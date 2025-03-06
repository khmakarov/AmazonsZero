# train.py
import os
import json
import zlib
import time
import torch
import torch.optim as optim
import numpy as np
import random
import multiprocessing
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from src.utils.database import AmazonsDatabase
from Amazons import GameCore

TOTAL_ACTIONS = 33344


class Trainer:

    def __init__(self, args):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.game = GameCore()
        self.nnet = self._init_net().to(self.device)
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr, weight_decay=1e-4)
        self.train_examples = []
        self.batch_size = args.batch_size
        self.num_iters = args.num_iters
        self.num_eps = args.num_eps
        self.checkpoint_dir = args.checkpoint_dir
        self.db = AmazonsDatabase()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _init_net(self):
        nnet = AlphaZeroNet().to(self.device)
        if self.args.load_model:
            checkpoint = torch.load(self.args.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def execute_episode(self):
        episode_data = []
        state = GameCore(self.game)
        mcts = MCTS(self.nnet, Args())
        while True:
            pi = mcts.getActionProb(state)
            action_index = np.random.choice(len(pi), p=pi)
            next_state = GameCore(state)
            next_state.step(action_index)
            ended = next_state.is_terminal()
            action = self.game.index2action(action_index)
            episode_data.append([state, action, pi])
            if ended != 0:
                episode_data.append([next_state, None, None])
                result = "黑胜" if ended == 1 and next_state.current_player == 0 else "白胜"
                print(self.db.save_game(episode_data, result))
                return [(x[0], x[2], ended) for x in episode_data[:-1]]
            state = next_state

    @staticmethod
    def _serialize_state(state) -> bytes:
        """序列化游戏状态为压缩字节流"""
        state_data = {"black": int(state.black), "white": int(state.white), "blocks": int(state.blocks), "current_player": state.current_player}
        return zlib.compress(json.dumps(state_data).encode())

    @staticmethod
    def _deserialize_state(data: bytes):
        """从字节流重建游戏状态"""
        state_data = json.loads(zlib.decompress(data).decode())

        # 重建GameCore实例
        state = GameCore()
        state.current_player = state_data["current_player"]
        state.black = state_data["black"]
        state.white = state_data["white"]
        state.blocks = state_data["blocks"]

        return state

    @staticmethod
    def execute_episode_worker(nnet_state_dict, args):
        # 子进程初始化模型和MCTS
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nnet = AlphaZeroNet().to(device)
        nnet.load_state_dict(nnet_state_dict)
        mcts = MCTS(nnet, args)

        episode_data = []
        state = GameCore()
        while True:
            pi = mcts.getActionProb(state)
            action_index = np.random.choice(len(pi), p=pi)
            next_state = GameCore(state)
            next_state.step(action_index)
            ended = next_state.is_terminal()
            action = state.index2action(action_index)
            print(action)
            episode_data.append([Trainer._serialize_state(state), action, pi])
            if ended != 0:
                result = "黑胜" if ended == 1 and next_state.current_player == 0 or ended == 0 and next_state.current_player == 1 else "白胜"
                return (  # 返回训练数据和原始数据
                    [(x[0], x[2], ended) for x in episode_data],  # 训练样本
                    episode_data + [[Trainer._serialize_state(next_state), None, None]],  # 完整对弈数据
                    result
                )
            state = next_state

    def learn(self):
        for i in range(1, self.num_iters + 1):
            print(f"Iteration {i}")

            nnet_state_dict = self.nnet.to("cpu").state_dict()
            with multiprocessing.Pool(processes=os.cpu_count()) as pool:
                tasks = [(nnet_state_dict, self.args)] * self.num_eps
                results = pool.starmap(Trainer.execute_episode_worker, tasks)

            eps_data = []
            for train_data, episode_data, result in results:
                eps_data.extend((self._deserialize_state(s_bytes), pi, ended) for s_bytes, pi, ended in train_data)
                episode_data_visual = [(self._deserialize_state(s_bytes), a, pi) for s_bytes, a, pi in episode_data]
                print(self.db.save_game(episode_data_visual, result))  # 主进程统一保存
            self.train_examples.extend(eps_data)
            if len(self.train_examples) >= self.batch_size:
                batch = random.sample(self.train_examples, self.batch_size)
                self.train(batch)
            if i % self.args.checkpoint_freq == 0:
                self.save_checkpoint(i)

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

        print(f"Loss: {total_loss.item()}, Policy Loss: {loss_pi.item()}, Value Loss: {loss_v.item()}")

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

    def save_checkpoint(self, iteration):
        filepath = os.path.join(self.checkpoint_dir, f'checkpoint_{iteration}.pth')
        torch.save({
            'model': self.nnet.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, filepath)
        print(f"Saved checkpoint to {filepath}")


class Args:
    numMCTSSims = 128  # MCTS模拟次数
    cpuct = 1.0  # 探索系数
    lr = 0.001  # 学习率
    batch_size = 256  # 训练批次大小
    num_iters = 3  # 训练总迭代次数
    num_eps = 8  # 每迭代自我对弈次数
    checkpoint_freq = 1  # 保存间隔
    checkpoint_dir = "../../model/checkpoint"
    load_model = None  # 预训练模型路径


if __name__ == "__main__":
    start_time = time.perf_counter()
    trainer = Trainer(Args())
    trainer.learn()
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"代码执行时间: {execution_time:.6f} 秒")
