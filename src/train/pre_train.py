import os
import h5py
import hydra
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from mpi4py import MPI
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from core.python.net import AlphaZeroNet
from src.utils.ckpt_manager import CheckpointManager
from src.utils.data_manager import DataManager
from evaluate import Evaluator
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, ConstantLR, SequentialLR


def configure_scheduler(optimizer, total_epochs, steps_per_epoch):
    # 阶段1：渐进式Warmup (5%)
    warmup_epochs = 2
    warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs * steps_per_epoch)

    # 阶段2：余弦退火主训练 (75%)
    cosine_epochs = 16
    cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs * steps_per_epoch, eta_min=5e-5)

    # 阶段3：平稳微调 (20%)
    fine_tune = ConstantLR(optimizer, factor=0.01, total_iters=int(0.1 * total_epochs) * steps_per_epoch)

    milestones = [warmup_epochs * steps_per_epoch, (warmup_epochs + cosine_epochs) * steps_per_epoch]

    return SequentialLR(optimizer, schedulers=[warmup, cosine, fine_tune], milestones=milestones)


class HDF5Dataset(Dataset):

    def __init__(self, file_path):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank == 0:
            with h5py.File(file_path, 'r') as f:
                total_samples = f['states'].shape[0]
                full_indices = np.arange(total_samples)
        else:
            full_indices = None

        full_indices = self.comm.bcast(full_indices, root=0)
        self.sample_indices = np.array_split(full_indices, self.size)[self.rank]
        self.length = len(self.sample_indices)

        with h5py.File(file_path, 'r') as h5_file:
            self.states = h5_file['states'][self.sample_indices].astype(np.float32)
            self.pis = h5_file['pis'][self.sample_indices].astype(np.float32)
            self.valids = h5_file['valids'][self.sample_indices].astype(np.int32)
            self.ended = h5_file['ended'][self.sample_indices].astype(np.float32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (self.states[idx], self.pis[idx], self.valids[idx], self.ended[idx])


class PreTrainer:

    def __init__(self, cfg):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.cfg = cfg
        self.data_paths = [f"/root/autodl-tmp/data/data{i}.h5" for i in range(1, 5)]  # 数据集路径列表
        self.device = torch.device("cuda")
        self.load_model = cfg.load_model
        self.training = cfg.training
        self.win_rate = cfg.evaluator.win_rate
        if self.rank == 0:
            self.nnet = self._init_net()
            self.optimizer = AdamW(self.nnet.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
            if self.load_model:
                print("LOAD MODEL")
                checkpoint = torch.load(self.load_model)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler = configure_scheduler(self.optimizer, total_epochs=self.training.epoch, steps_per_epoch=352)
            self.ckpt_mgr = CheckpointManager()
            self.data_mgr = DataManager()
            self.evaluator = Evaluator(cfg)
            self.writer = SummaryWriter(cfg.log_dir)
            self.iteration = 0
            self.batch = 0
            self.color_palette = {'loss': '#1F77B4', 'policy': '#FF7F0E', 'value': '#2CA02C', 'lr': '#D62728', 'gradient': '#9467BD', 'attention': '#8C564B'}
            self.gradient_history = []
            self.se_activations = []
            self.topk_history = {'Top-1': [], 'Top-5': [], 'Top-10': []}

    def _init_net(self):
        nnet = AlphaZeroNet().to(self.device)
        if self.load_model:
            checkpoint = torch.load(self.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def _log_gradients(self):
        """全网络梯度分布监控（残差块/策略头/价值头）"""
        gradient_data = {'res_blocks': [], 'policy_head': [], 'value_head': []}

        # 1. 遍历所有参数并分类记录梯度
        for name, param in self.nnet.named_parameters():
            if param.grad is None:
                continue  # 跳过无梯度参数

            # 提取梯度统计量
            grad_norm = torch.norm(param.grad).item()  # L2范数
            grad_mean = param.grad.abs().mean().item()  # 绝对值均值
            grad_max = param.grad.abs().max().item()  # 最大绝对值

            # 按模块分类
            if 'res_blocks' in name:
                block_idx = int(name.split('.')[1])  # 获取残差块索引
                if len(gradient_data['res_blocks']) <= block_idx:
                    gradient_data['res_blocks'].append({})
                layer_type = 'conv' if 'conv' in name else 'bn'
                gradient_data['res_blocks'][block_idx][f'{layer_type}_grad'] = (grad_norm, grad_mean, grad_max)
            elif 'policy' in name:
                gradient_data['policy_head'].append((grad_norm, grad_mean, grad_max))
            elif 'value' in name:  # 新增价值头参数判断
                gradient_data['value_head'].append((grad_norm, grad_mean, grad_max))

        # 2. 可视化残差块梯度分布（按层）
        fig_res = self._plot_resblock_gradients(gradient_data['res_blocks'])
        self.writer.add_figure('Gradients/ResBlocks', fig_res, self.batch)

        # 3. 可视化策略头梯度分布
        fig_policy = self._plot_head_gradients(gradient_data['policy_head'], 'Policy Head')
        self.writer.add_figure('Gradients/PolicyHead', fig_policy, self.batch)
        # 4. 新增价值头梯度分布可视化
        fig_value = self._plot_head_gradients(gradient_data['value_head'], 'Value Head')
        self.writer.add_figure('Gradients/ValueHead', fig_value, self.batch)
        # 5. 记录标量统计量（用于趋势分析）
        self._log_scalar_metrics(gradient_data)

    def _plot_resblock_gradients(self, res_grads):
        """残差块梯度热力图（区分conv/bn层）"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))

        # 绘制卷积层梯度
        conv_data = [block.get('conv_grad', (0, 0, 0))[0] for block in res_grads]
        im1 = axes[0].imshow([conv_data], cmap='viridis', aspect='auto')
        axes[0].set_title("Conv Layers Gradient Norm")
        plt.colorbar(im1, ax=axes[0])

        # 绘制BN层梯度
        bn_data = [block.get('bn_grad', (0, 0, 0))[0] for block in res_grads]
        im2 = axes[1].imshow([bn_data], cmap='plasma', aspect='auto')
        axes[1].set_title("BN Layers Gradient Norm")
        plt.colorbar(im2, ax=axes[1])

        plt.tight_layout()
        return fig

    def _plot_head_gradients(self, grad_list, title):
        """策略头/价值头梯度分布直方图"""
        fig, ax = plt.subplots(figsize=(10, 4))

        if grad_list:
            # 根据头部类型选择颜色
            color = '#FF7F0E' if 'Policy' in title else '#2CA02C'

            # 提取各层L2范数
            norms = [g[0] for g in grad_list]
            layers = [f'Layer {i}' for i in range(len(norms))]

            # 绘制柱状图
            ax.bar(layers, norms, alpha=0.6, color=color)
            ax.set_yscale('log')  # 对数尺度显示大范围变化
            ax.set_title(f"{title} Gradient Norms (Log Scale)")
            plt.xticks(rotation=45)

        return fig

    def _log_scalar_metrics(self, data):
        """记录关键梯度统计指标（趋势分析）"""
        # 残差块全局指标
        res_conv_grads = [b['conv_grad'][0] for b in data['res_blocks'] if 'conv_grad' in b]
        res_bn_grads = [b['bn_grad'][0] for b in data['res_blocks'] if 'bn_grad' in b]

        if res_conv_grads:
            self.writer.add_scalar('GradStats/ResBlock_Conv_Mean', np.mean(res_conv_grads), self.batch)
            self.writer.add_scalar('GradStats/ResBlock_BN_Mean', np.mean(res_bn_grads), self.batch)

        # 策略头梯度统计
        if data['policy_head']:
            policy_norms = [g[0] for g in data['policy_head']]
            self.writer.add_scalar('GradStats/PolicyHead_Max', max(policy_norms), self.batch)
            self.writer.add_scalar('GradStats/PolicyHead_Mean', np.mean(policy_norms), self.batch)
        if data['value_head']:
            value_norms = [g[0] for g in data['value_head']]
            self.writer.add_scalar('GradStats/ValueHead_Max', max(value_norms), self.batch)
            self.writer.add_scalar('GradStats/ValueHead_Mean', np.mean(value_norms), self.batch)

    def _log_policy_distribution(self, out_pi, valids):
        probs = torch.exp(out_pi).cpu().detach().numpy()
        valids = valids.cpu().numpy()
        valid_probs = probs[valids]
        # 计算批次平均Top-K
        topk_batch_avg = []
        for k in [1, 5, 10]:
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            batch_avg = np.mean(sorted_probs[:, :k].sum(axis=1))
            topk_batch_avg.append(batch_avg)

        # 更新历史记录
        self.topk_history['Top-1'].append(topk_batch_avg[0])
        self.topk_history['Top-5'].append(topk_batch_avg[1])
        self.topk_history['Top-10'].append(topk_batch_avg[2])

        # 创建趋势图表
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # 趋势曲线（修正横坐标）
        iterations = list(range(len(self.topk_history['Top-1'])))
        for k, key in zip([1, 5, 10], ['Top-1', 'Top-5', 'Top-10']):
            ax1.plot(iterations, self.topk_history[key], label=f'{key}')
        ax1.set_xlabel('Training Iteration')
        ax1.set_ylabel('Average Probability')
        ax1.legend()

        bins = np.logspace(-5, 0, 50, base=10)
        ax2.hist(valid_probs, bins=bins, color='royalblue', alpha=0.8, edgecolor='black', linewidth=0.5)

        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.set_xlabel('Action Probability (Log Scale)')
        ax2.set_ylabel('Count (Log Scale)')

        action_space_size = valids.shape[1]
        random_level = 1 / action_space_size

        # 优化后的参考线标注
        ax2.axvline(random_level, color='crimson', linestyle=':', linewidth=1.2, alpha=0.7)
        ax2.text(
            x=random_level * 2,  # 向右偏移一个数量级
            y=0.9 * ax2.get_ylim()[1],
            s=f' Random Level\n({random_level:.1e})',
            rotation=0,
            fontsize=7,
            color='crimson',
            va='top',
            ha='left',
            transform=ax2.transData,
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='crimson', boxstyle='round,pad=0.2')
        )

        stats_text = (
            f"Valid Actions: {len(valid_probs):,}\n"
            f"Min: {valid_probs.min():.1e}\n"
            f"Max: {valid_probs.max():.3f}\n"
            f"Mean: {valid_probs.mean():.1e}\n"
            f"95th%: {np.percentile(valid_probs, 95):.1e}"
        )
        ax2.text(0.65, 0.75, stats_text, transform=ax2.transAxes, fontsize=8, bbox=dict(facecolor='white', alpha=0.9))

        plt.tight_layout()
        self.writer.add_figure('Policy/Distribution', fig, self.iteration)
        plt.close()

    def train_all_datasets(self):
        for i in range(self.training.epoch):
            if self.rank == 0:
                print(f"Epoch:{i+1}/{self.training.epoch}")
            for idx, data_path in enumerate(self.data_paths):
                if self.rank == 0:
                    print(f"\n=== Training on Dataset {idx+1}/{len(self.data_paths)}: {data_path} ===")
                self._process_dataset(data_path)
                if self.rank == 0:
                    self.train()
                    del self.full_dataset, self.train_loader

                self.comm.barrier()
            if self.rank == 0:
                self.ckpt_mgr.save(self.nnet, self.optimizer, self.scheduler)
                print(f"Checkpoint saved after {data_path}")

    def _process_dataset(self, file_path):
        dataset = HDF5Dataset(file_path=file_path)
        if self.rank == 0:
            all_states = [torch.from_numpy(dataset.states).float().permute(0, 3, 1, 2)]
            all_pis = [torch.from_numpy(dataset.pis).float()]
            all_valids = [torch.from_numpy(dataset.valids)]
            all_ended = [torch.from_numpy(dataset.ended).float()]

            for src_rank in range(1, self.size):
                shape_info = self.comm.recv(source=src_rank, tag=4)

                states = np.empty(shape_info['states'], dtype=np.float32)
                self.comm.Recv(states, source=src_rank, tag=0)
                all_states.append(torch.from_numpy(states).float().permute(0, 3, 1, 2))

                pis = np.empty(shape_info['pis'], dtype=np.float32)
                self.comm.Recv(pis, source=src_rank, tag=1)
                all_pis.append(torch.from_numpy(pis).float())

                valids = np.empty(shape_info['pis'][:-1] + (1500, ), dtype=np.int32)
                self.comm.Recv(valids, source=src_rank, tag=2)
                all_valids.append(torch.from_numpy(valids))

                ended = np.empty(shape_info['pis'][:-1], dtype=np.float32)
                self.comm.Recv(ended, source=src_rank, tag=3)
                all_ended.append(torch.from_numpy(ended).float())

            self.full_dataset = torch.utils.data.TensorDataset(
                torch.cat(all_states).contiguous(),
                torch.cat(all_pis).contiguous(),
                torch.cat(all_valids).contiguous(),
                torch.cat(all_ended).contiguous()
            )
            self.train_loader = DataLoader(
                self.full_dataset, batch_size=self.training.batch_size, pin_memory=True, shuffle=True, num_workers=2, persistent_workers=True, prefetch_factor=2
            )
        else:
            shape_info = {'states': dataset.states.shape, 'pis': dataset.pis.shape}
            self.comm.send(shape_info, dest=0, tag=4)

            self.comm.Send(np.ascontiguousarray(dataset.states), dest=0, tag=0)
            self.comm.Send(np.ascontiguousarray(dataset.pis), dest=0, tag=1)
            self.comm.Send(np.ascontiguousarray(dataset.valids), dest=0, tag=2)
            self.comm.Send(np.ascontiguousarray(dataset.ended), dest=0, tag=3)

        self.comm.barrier()

    def train(self):
        self.nnet.train()
        total_loss_sum = 0.0
        total_pi_sum = 0.0
        total_v_sum = 0.0

        for batch_idx, (states, pis, valids, ended) in enumerate(self.train_loader):
            states = states.to(self.device, non_blocking=True)
            pis = pis.to(self.device, non_blocking=True)
            valids = valids.to(self.device, non_blocking=True)
            ended = ended.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            out_pi, out_v, valids = self.nnet(states, valids)
            loss_pi = -torch.sum(pis * out_pi) / pis.size(0)
            loss_v = F.mse_loss(out_v, ended.unsqueeze(1))
            total_loss = loss_pi + loss_v
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.nnet.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss_sum += total_loss.item() * states.size(0)
            total_pi_sum += loss_pi.item() * states.size(0)
            total_v_sum += loss_v.item() * states.size(0)

            if batch_idx % 10 == 0:
                self._log_gradients()
                self._log_policy_distribution(out_pi, valids)
                self.batch += 1
                print(f"Batch {batch_idx}/{len(self.train_loader)}: "
                      f"Loss={total_loss.item():.8f} "
                      f"(PI={loss_pi.item():.8f} V={loss_v.item():.8f})")

        avg_loss = total_loss_sum / len(self.train_loader.dataset)
        avg_pi = total_pi_sum / len(self.train_loader.dataset)
        avg_v = total_v_sum / len(self.train_loader.dataset)

        self.writer.add_scalar('Loss/total', avg_loss, self.iteration)
        self.writer.add_scalar('Loss/pi', avg_pi, self.iteration)
        self.writer.add_scalar('Loss/v', avg_v, self.iteration)
        self.writer.add_scalar('Learning Rate', self.optimizer.param_groups[0]['lr'], self.iteration, new_style=True)
        self.iteration += 1


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg) -> None:
    # 初始化MPI环境
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    trainer = PreTrainer(cfg)
    trainer.train_all_datasets()


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    main()
