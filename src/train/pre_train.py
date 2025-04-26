import h5py
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import os
from mpi4py import MPI
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from core.python.net import AlphaZeroNet
from src.utils.ckpt_manager import CheckpointManager
from src.utils.data_manager import DataManager
from evaluate import Evaluator
from torch.utils.tensorboard.writer import SummaryWriter


class HDF5Dataset(Dataset):

    def __init__(self, file_path, ratio):

        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank == 0:
            with h5py.File(file_path, 'r') as f:
                total_samples = f['states'].shape[0]
            if ratio == 1.0:
                full_indices = np.arange(total_samples)
            else:
                full_indices = np.random.choice(total_samples, int(total_samples * ratio), replace=False)
                full_indices.sort()
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
        if self.rank == 0:
            self.device = torch.device("cuda")
            self.load_model = cfg.load_model
            self.nnet = self._init_net()
            self.training = cfg.training
            self.win_rate = cfg.evaluator.win_rate
            self.optimizer = Adam(self.nnet.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
            if self.load_model:
                print("LOAD MODEL")
                checkpoint = torch.load(self.load_model)
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.ckpt_mgr = CheckpointManager()
            self.data_mgr = DataManager()
            self.evaluator = Evaluator(cfg)
            self.writer = SummaryWriter(cfg.log_dir)
            self.iteration = 0
            self.writer.add_graph(self.nnet, torch.randn(1, 5, 8, 8).to(self.device))

        self.dataset = HDF5Dataset("/home/khmakarov/AmazonsZero/data/pretrain/data1.h5", ratio=0.01)

        if self.rank == 0:
            all_states = [torch.from_numpy(self.dataset.states).float().permute(0, 3, 1, 2)]
            all_pis = [torch.from_numpy(self.dataset.pis).float()]
            all_valids = [torch.from_numpy(self.dataset.valids)]
            all_ended = [torch.from_numpy(self.dataset.ended).float()]

            for src_rank in range(1, self.comm.Get_size()):
                shape_info = self.comm.recv(source=src_rank, tag=4)
                states_shape = shape_info['states']
                pis_shape = shape_info['pis']

                states = np.empty(states_shape, dtype=np.float32)
                self.comm.Recv(states, source=src_rank, tag=0)
                all_states.append(torch.from_numpy(states).float().permute(0, 3, 1, 2))

                pis = np.empty(pis_shape, dtype=np.float32)
                self.comm.Recv(pis, source=src_rank, tag=1)
                all_pis.append(torch.from_numpy(pis).float())

                valids = np.empty(pis_shape[:-1] + (1500, ), dtype=np.int32)
                self.comm.Recv(valids, source=src_rank, tag=2)
                all_valids.append(torch.from_numpy(valids))

                ended = np.empty(pis_shape[:-1], dtype=np.float32)
                self.comm.Recv(ended, source=src_rank, tag=3)
                all_ended.append(torch.from_numpy(ended).float())

            self.comm.barrier()
            self.dataset.states = None
            self.dataset.pis = None
            self.dataset.valids = None
            self.dataset.ended = None
            self.full_dataset = torch.utils.data.TensorDataset(torch.cat(all_states), torch.cat(all_pis), torch.cat(all_valids), torch.cat(all_ended))
            self.train_loader = DataLoader(self.full_dataset, batch_size=cfg.training.batch_size, pin_memory=True, shuffle=True)

            print("\n===== 数据集诊断信息 =====")

            # 查看合并后的总样本数
            print(f"总样本数: {len(self.full_dataset)}")

            # 查看单个样本结构
            sample = self.full_dataset[0]
            print("\n单个样本结构:")
            print(f"states shape: {sample[0].shape} | dtype: {sample[0].dtype}")
            print(f"pis shape:    {sample[1].shape} | dtype: {sample[1].dtype}")
            print(f"valids shape: {sample[2].shape} | dtype: {sample[2].dtype}")
            print(f"ended shape:  {sample[3].shape} | dtype: {sample[3].dtype}")

            # 查看数据范围
            print("\n数据范围:")
            print(f"states min/max: {torch.min(sample[0])}, {torch.max(sample[0])}")
            print(f"pis sum check:  {torch.sum(sample[1])} (应为 1.0)")
            print(f"ended values:   {torch.unique(self.full_dataset.tensors[3])}")

            # 查看各进程数据量
            print("\n各进程加载数据量:")
            for i, (s, p, v, e) in enumerate(zip(all_states, all_pis, all_valids, all_ended)):
                print(f"Rank {i}: states={len(s)}, pis={len(p)}, valids={len(v)}, ended={len(e)}")

            print("==========================\n")
        else:
            states = np.ascontiguousarray(self.dataset.states)
            pis = np.ascontiguousarray(self.dataset.pis)
            valids = np.ascontiguousarray(self.dataset.valids)
            ended = np.ascontiguousarray(self.dataset.ended)
            shape_info = {'states': self.dataset.states.shape, 'pis': self.dataset.pis.shape}
            self.comm.send(shape_info, dest=0, tag=4)
            self.comm.Send(self.dataset.states, dest=0, tag=0)
            self.comm.Send(self.dataset.pis, dest=0, tag=1)
            self.comm.Send(self.dataset.valids, dest=0, tag=2)
            self.comm.Send(self.dataset.ended, dest=0, tag=3)
            self.comm.barrier()
            exit(0)

    def _init_net(self):
        nnet = AlphaZeroNet().to(self.device)
        if self.load_model:
            checkpoint = torch.load(self.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def train_epoch(self):
        if self.rank == 0:
            self.nnet.train()
            stream = torch.cuda.Stream()  # 创建专用 CUDA 流
            next_batch = None

            for epoch in range(self.training.epoch):
                total_loss_sum = 0.0
                total_pi_sum = 0.0
                total_v_sum = 0.0

                data_iter = iter(self.train_loader)
                try:
                    current_batch = next(data_iter)
                except StopIteration:
                    break

                with torch.cuda.stream(stream):
                    next_batch = next(data_iter, None)
                    if next_batch is not None:
                        next_batch = [t.to(self.device, non_blocking=True) for t in next_batch]

                while True:
                    current_states, current_pis, current_valids, current_target_v = [t.to(self.device, non_blocking=True) for t in current_batch]
                    torch.cuda.current_stream().wait_stream(stream)

                    self.optimizer.zero_grad()
                    out_pi, out_v = self.nnet(current_states, current_valids)
                    loss_pi = -torch.sum(current_pis * out_pi) / current_pis.size(0)
                    loss_v = F.mse_loss(out_v, current_target_v.unsqueeze(1))
                    batch_total_loss = loss_pi + loss_v
                    batch_total_loss.backward()
                    self.optimizer.step()

                    batch_size = current_states.size(0)
                    total_loss_sum += batch_total_loss.item() * batch_size
                    total_pi_sum += loss_pi.item() * batch_size
                    total_v_sum += loss_v.item() * batch_size

                    # 预加载下一个 batch（在另一个流中）
                    if next_batch is not None:
                        current_batch = next_batch
                        with torch.cuda.stream(stream):
                            try:
                                next_batch = next(data_iter)
                                next_batch = [t.to(self.device, non_blocking=True) for t in next_batch]
                            except StopIteration:
                                next_batch = None
                    else:
                        break

                avg_loss = total_loss_sum / len(self.train_loader.dataset)
                avg_pi = total_pi_sum / len(self.train_loader.dataset)
                avg_v = total_v_sum / len(self.train_loader.dataset)

                print(f"Epoch {epoch+1}/{self.training.epoch}")
                print(f"Total Loss: {avg_loss:.8f} | PI Loss: {avg_pi:.8f} | V Loss: {avg_v:.8f}")
                self.writer.add_scalar('Loss/total', avg_loss, epoch)
                self.writer.add_scalar('Loss/pi', avg_pi, epoch)
                self.writer.add_scalar('Loss/v', avg_v, epoch)

                if (epoch + 1) % 10 == 0:
                    self.ckpt_mgr.save(self.nnet, self.optimizer)

    def _evaluation_step(self):
        latest_path = "/home/khmakarov/AmazonsZero/model/checkpoint/checkpoint_20250425_212024_nan.pth"  #self.ckpt_mgr.get_latest_checkpoint()
        current_path = "/home/khmakarov/AmazonsZero/model/checkpoint/checkpoint_20250425_205931_nan.pth"  #self.ckpt_mgr.save(self.nnet, self.optimizer)

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


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg) -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        trainer = PreTrainer(cfg)
        trainer._evaluation_step()
    else:
        trainer = PreTrainer(cfg)


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    mp.set_start_method('spawn', force=True)
    main()
