# train.py
import os
import pstats
import hydra
import cProfile
import numpy as np
import torch
import torch.multiprocessing as mp
from matplotlib import pyplot as plt
from torch.optim import AdamW
from core.cpp.build.Amazons import GameCore
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from src.utils.ckpt_manager import CheckpointManager
from src.utils.data_manager import DataManager
from evaluate import Evaluator
from torch.utils.tensorboard.writer import SummaryWriter


class Trainer:

    def __init__(self, cfg):
        self.device = torch.device("cuda")
        self.load_model = cfg.load_model
        self.mcts = cfg.mcts
        self.nnet = self._init_net()
        self.training = cfg.training
        self.win_rate = cfg.evaluator.win_rate
        self.optimizer = AdamW(self.nnet.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        if self.load_model:
            checkpoint = torch.load(self.load_model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ckpt_mgr = CheckpointManager()
        self.data_mgr = DataManager()
        self.evaluator = Evaluator(cfg)
        self.writer = SummaryWriter(cfg.log_dir)
        self.iteration = 0
        self.color_palette = {'loss': '#1F77B4', 'policy': '#FF7F0E', 'value': '#2CA02C', 'lr': '#D62728', 'gradient': '#9467BD', 'attention': '#8C564B'}
        self.gradient_history = []
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
            elif 'value' in name:
                gradient_data['value_head'].append((grad_norm, grad_mean, grad_max))

        # 2. 可视化残差块梯度分布（按层）
        fig_res = self._plot_resblock_gradients(gradient_data['res_blocks'])
        self.writer.add_figure('Gradients/ResBlocks', fig_res, self.iteration)

        # 3. 可视化策略头梯度分布
        fig_policy = self._plot_head_gradients(gradient_data['policy_head'], 'Policy Head')
        self.writer.add_figure('Gradients/PolicyHead', fig_policy, self.iteration)

        fig_value = self._plot_head_gradients(gradient_data['value_head'], 'Value Head')
        self.writer.add_figure('Gradients/ValueHead', fig_value, self.iteration)
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
            # 提取各层L2范数
            norms = [g[0] for g in grad_list]
            layers = [f'Layer {i}' for i in range(len(norms))]

            # 绘制柱状图
            ax.bar(layers, norms, alpha=0.6)
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
            self.writer.add_scalar('GradStats/ResBlock_Conv_Mean', np.mean(res_conv_grads), self.iteration)
            self.writer.add_scalar('GradStats/ResBlock_BN_Mean', np.mean(res_bn_grads), self.iteration)

        # 策略头梯度统计
        if data['policy_head']:
            policy_norms = [g[0] for g in data['policy_head']]
            self.writer.add_scalar('GradStats/PolicyHead_Max', max(policy_norms), self.iteration)

    def _log_policy_distribution(self, out_pi, valids):
        probs = torch.exp(out_pi).cpu().detach().numpy()
        valids = valids.cpu().numpy()
        valid_probs = probs[valids]
        # 计算批次平均Top-K
        topk_batch_avg = []
        for k in [1, 5, 10]:
            sorted_probs = np.sort(probs, axis=1)[:, ::-1]
            batch_avg = np.mean(sorted_probs[:, :k].sum(axis=1))  # 关键修改：先求和再平均
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

    def learn(self):
        for i in range(1, self.training.num_iters + 1):
            print(f"Iteration {i}")
            self.iteration = i
            self._process_episodes(self._generate_episodes())
            self.data_mgr.flush_visual_data()
            self._train_step()
            if i % self.training.eval_freq == 0:
                self._evaluation_step()

    def _generate_episodes(self):
        self.nnet.share_memory()
        self.nnet.eval()
        with mp.Pool(processes=1) as pool:
            tasks = [(self.nnet, self.mcts)] * self.training.num_eps
            results = pool.starmap(self.execute_episode_worker, tasks)
            pool.close()
            pool.join()
        return results

    def _process_episodes(self, results):
        for episode_data in results:
            train_data, visual_data = zip(*[((x[0], x[1], x[2], x[3]), (x[0], x[4])) for x in episode_data])
            self.data_mgr.add_train_data(train_data)
            self.data_mgr.add_visual_data(visual_data, episode_data[-1][3], self.iteration, 0)
        print("该迭代数据保存完毕")

    def _train_step(self):
        batches = self.data_mgr.sample_batch(self.training.batch_size)
        for batch in batches:
            total_loss, loss_pi, loss_v = self.train(batch)

            print(f"Loss={total_loss:.8f} ", f"Loss/policy={loss_pi:.8f} ", f"Loss/value={loss_v:.8f} ")
            self.writer.add_scalar('Loss/total', total_loss, self.iteration)
            self.writer.add_scalar('Loss/policy', loss_pi, self.iteration)
            self.writer.add_scalar('Loss/value', loss_v, self.iteration)
        print("该迭代数据训练完毕")

    def _evaluation_step(self):
        latest_path = self.ckpt_mgr.get_latest_checkpoint()
        current_path = self.ckpt_mgr.save(self.nnet, self.optimizer)

        if not latest_path:
            print(f"Initial model saved: {current_path}")
            return
        win_rate = self.evaluator.evaluate(self.iteration, current_path, latest_path)

        if win_rate >= self.win_rate:
            best_model_path = self.ckpt_mgr.save(self.nnet, self.optimizer, win_rate)
            self.data_mgr.clear_train_data()
            print(f"New best model: {best_model_path} (Win rate: {win_rate:.2f})")
        else:
            del self.nnet
            self.nnet = self._init_net()
            best_model_path = latest_path
            checkpoint = torch.load(best_model_path)
            self.nnet.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Current model rejected. Win rate: {win_rate:.2f}")

        self.writer.add_scalar('Evaluation/win_rate', win_rate, self.iteration)

    def train(self, batch):
        self.nnet.train()

        states, pis, valids_idx, vs = list(zip(*batch))
        states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        states = states.permute(0, 3, 1, 2).to(self.device)
        target_pis = torch.tensor(np.array(pis), dtype=torch.float32).to(self.device)
        target_vs = torch.tensor(np.array(vs), dtype=torch.float32).unsqueeze(1).to(self.device)
        valids_idx = torch.as_tensor(np.stack(valids_idx), dtype=torch.int, device=self.device)

        out_pi, out_v, valids = self.nnet(states, valids_idx)
        loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
        loss_v = torch.mean((target_vs - out_v)**2)
        total_loss = loss_pi + loss_v

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        self._log_gradients()
        self._log_policy_distribution(out_pi, valids)
        for name, param in self.nnet.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, self.iteration)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, self.iteration)

        return total_loss.detach().item(), loss_pi.detach().item(), loss_v.detach().item()

    @staticmethod
    def execute_episode_worker(nnet, cfg):
        try:
            episode_data = []
            mcts = MCTS(nnet, cfg)
            state = GameCore()
            pr = cProfile.Profile()
            pr.enable()
            while True:
                ended = state.is_terminal()
                if ended == 0:
                    pi, valids_idx, v = mcts.getActionProb(state)
                    action_index = np.argmax(pi)
                    next_state = GameCore(state)
                    next_state.step(action_index)
                    episode_data.append([state.get_state_np(), pi, valids_idx, v, state.index2action(action_index)])
                    state = next_state
                else:
                    episode_data.append(
                        [state.get_state_np(),
                         np.zeros(cfg.TOTAL_ACTIONS, dtype=np.float16),
                         np.zeros(cfg.POSSIBLE_ACTIONS, dtype=np.int32), ended,
                         state.index2action(0)]
                    )
                    break
            return episode_data
        finally:
            pr.disable()
            stats = pstats.Stats(pr)
            stats.strip_dirs()  # 移除文件路径前缀
            stats.sort_stats('cumtime')  # 按累计时间排序
            stats.print_stats(r'mcts\.py|net\.py|train\.py')  # 仅显示指定文件
            del mcts, nnet
            torch.cuda.empty_cache()

    def __del__(self):
        self.writer.close()
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg) -> None:
    trainer = Trainer(cfg)
    trainer.learn()


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    mp.set_start_method('spawn', force=True)
    main()
