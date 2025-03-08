# train.py
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import zlib
import torch
import numpy as np
import multiprocessing
import hydra
from Amazons import GameCore
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from evaluate import Evaluator
from src.utils.ckpt_manager import CheckpointManager
from src.utils.data_manager import DataManager
from torch.utils.tensorboard import SummaryWriter


class Trainer:

    def __init__(self, cfg):
        # 初始化基础组件
        self.device = torch.device("cuda")
        self.game = GameCore()
        self.training = cfg.training
        self.mcts = cfg.mcts
        self.load_model = cfg.load_model
        self.win_rate = cfg.evaluator.win_rat
        # 模块化组件
        self.nnet = self._init_net()
        self.optimizer = torch.optim.Adam(self.nnet.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        self.ckpt_mgr = CheckpointManager()
        self.data_mgr = DataManager()
        self.evaluator = Evaluator(cfg)
        self.writer = SummaryWriter(cfg.log_dir)

        self.iteration = 0
        self.writer.add_graph(self.nnet, torch.randn(1, 5, 8, 8).to(self.device))

    def _init_net(self):
        nnet = AlphaZeroNet().to(self.device)
        if self.load_model:
            checkpoint = torch.load(self.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def learn(self):
        for i in range(1, self.training.num_iters + 1):
            print(f"Iteration {i}")
            self.iteration = i

            episodes = self._generate_episodes()
            self._process_episodes(episodes)

            if len(self.data_mgr.train_data) >= self.training.batch_size:
                self._train_step()

            if i % self.training.eval_freq == 0 and i > 1:
                self._evaluation_step()

    def _generate_episodes(self):
        nnet_state_dict = self.nnet.to("cpu").state_dict()
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            tasks = [(nnet_state_dict, self.mcts)] * self.training.num_eps
            return pool.starmap(self.execute_episode_worker, tasks)

    def _process_episodes(self, results):
        for train_data, episode_data, result in results:
            self.data_mgr.add_episode_data(episode_data, result)
            self.data_mgr.add_train_data(train_data)

    def _train_step(self):
        batch = self.data_mgr.sample_batch(self.training.batch_size)
        total_loss, loss_pi, loss_v = self.train(batch)

        self.writer.add_scalar('Loss/total', total_loss, self.iteration)
        self.writer.add_scalar('Loss/policy', loss_pi, self.iteration)
        self.writer.add_scalar('Loss/value', loss_v, self.iteration)

    def _evaluation_step(self):
        latest_path = self.ckpt_mgr.get_latest_checkpoint()
        current_path = self.ckpt_mgr.save(self.nnet, self.optimizer)

        if not latest_path:
            print(f"Initial model saved: {current_path}")
            return

        win_rate = self.evaluator.evaluate_models(current_path, latest_path)
        os.remove(current_path)

        if win_rate > self.win_rate:
            best_model_path = self.ckpt_mgr.save(self.nnet, self.optimizer, win_rate)
            self.data_mgr.clear_train_data()
            print(f"New best model: {best_model_path} (Win rate: {win_rate:.2f})")
        else:
            best_model_path = latest_path
            print(f"Current model rejected. Win rate: {win_rate:.2f}")

        self.writer.add_scalar('Evaluation/win_rate', win_rate, self.iteration)
        checkpoint = torch.load(best_model_path)
        self.nnet.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

    def train(self, batch):
        self.nnet.to(self.device)
        self.nnet.train()
        states, pis, vs = list(zip(*batch))

        states = torch.tensor(np.array([s.get_state() for s in states]), dtype=torch.float32)
        states = states.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        target_pis = torch.tensor(np.array(pis), dtype=torch.float32).to(self.device)
        target_vs = torch.tensor(np.array(vs), dtype=torch.float32).unsqueeze(1).to(self.device)

        out_pi, out_v = self.nnet(states)
        loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
        loss_v = torch.mean((target_vs - out_v)**2)
        total_loss = loss_pi + loss_v

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        for name, param in self.nnet.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, self.iteration)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, self.iteration)

        return total_loss.item(), loss_pi.item(), loss_v.item()

    @staticmethod
    def _serialize_state(state) -> bytes:
        """序列化游戏状态为压缩字节流"""
        state_data = {"black": int(state.black), "white": int(state.white), "blocks": int(state.blocks), "current_player": state.current_player}
        return zlib.compress(json.dumps(state_data).encode())

    @staticmethod
    def execute_episode_worker(nnet_state_dict, cfg):
        try:
            # 子进程初始化模型和MCTS
            device = torch.device("cuda")
            nnet = AlphaZeroNet().to(device)
            nnet.load_state_dict(nnet_state_dict)
            mcts = MCTS(nnet, cfg)

            episode_data = []
            state = GameCore()
            while True:
                pi = mcts.getActionProb(state)
                action_index = np.random.choice(len(pi), p=pi)
                next_state = GameCore(state)
                next_state.step(action_index)
                ended = next_state.is_terminal()
                action = state.index2action(action_index)
                episode_data.append([Trainer._serialize_state(state), action, pi])
                if ended != 0:
                    result = "黑胜" if (ended == 1 and next_state.current_player == 0) or (ended == 0 and next_state.current_player == 1) else "白胜"
                    return (  # 返回训练数据和原始数据
                        [(x[0], x[2], ended) for x in episode_data],  # 训练样本
                        episode_data + [[Trainer._serialize_state(next_state), None, None]],  # 完整对弈数据
                        result
                    )
                state = next_state
        finally:
            torch.cuda.empty_cache()

    def __del__(self):
        self.writer.close()


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg) -> None:
    trainer = Trainer(cfg)
    trainer.learn()


if __name__ == "__main__":
    main()
