# train.py
import os
import torch
import torch.optim as optim
import numpy as np
import random
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from src.utils.database import AmazonsDatabase
from Amazons import GameCore

TOTAL_ACTIONS = 33344


class Trainer:

    def __init__(self, args):
        self.args = args
        self.game = GameCore()
        self.nnet = self._init_net()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        self.mcts = MCTS(self.nnet, args)
        self.train_examples = []
        self.batch_size = args.batch_size
        self.num_iters = args.num_iters
        self.num_eps = args.num_eps
        self.checkpoint_dir = args.checkpoint_dir
        self.db = AmazonsDatabase()
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _init_net(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nnet = AlphaZeroNet().to(device)
        if self.args.load_model:
            checkpoint = torch.load(self.args.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def execute_episode(self):
        examples = []
        state = GameCore(self.game)
        while True:
            pi = self.mcts.getActionProb(state)
            examples.append([state, pi, None])
            action_index = np.random.choice(len(pi), p=pi)
            next_state = GameCore(state)
            next_state.step(action_index)
            ended = next_state.is_terminal()
            if ended != 0:
                return [(x[0], x[1], ended) for x in examples]
            state = next_state

    def execute_episode_visual(self):
        examples = []
        episode_data = []
        state = GameCore(self.game)
        while True:
            pi = self.mcts.getActionProb(state)
            examples.append([state, pi, None])

            action_index = np.random.choice(len(pi), p=pi)
            next_state = GameCore(state)
            next_state.step(action_index)
            ended = next_state.is_terminal()
            action = self.game.index2action(action_index)
            episode_data.append([state, action])
            if ended != 0:
                episode_data.append([next_state, None])
                result = "黑胜" if ended == 1 else "白胜"
                game_id = self.db.save_game(episode_data, result)
                print(game_id)
                return [(x[0], x[1], ended) for x in examples]
            state = next_state

    def learn(self):
        for i in range(1, self.num_iters + 1):
            print(f"Iteration {i}")
            # 自我对弈生成数据
            eps_data = []
            for _ in range(self.num_eps):
                eps_data += self.execute_episode_visual()
            self.train_examples.extend(eps_data)

            # 训练神经网络
            if len(self.train_examples) >= self.batch_size:
                batch = random.sample(self.train_examples, self.batch_size)
                self.train(batch)

            # 保存检查点
            if i % self.args.checkpoint_freq == 0:
                self.save_checkpoint(i)

    def train(self, batch):
        self.nnet.train()
        states, pis, vs = list(zip(*batch))
        states = torch.tensor(np.array([s.get_state() for s in states]), dtype=torch.float32).permute(0, 3, 1, 2)  # NHWC -> NCHW
        target_pis = torch.tensor(np.array(pis), dtype=torch.float32)
        target_vs = torch.tensor(np.array(vs), dtype=torch.float32).unsqueeze(1)

        # 前向计算
        out_pi, out_v = self.nnet(states)
        loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
        loss_v = torch.mean((target_vs - out_v)**2)
        total_loss = loss_pi + loss_v

        # 反向传播
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
    batch_size = 2048  # 训练批次大小
    num_iters = 3  # 训练总迭代次数
    num_eps = 1  # 每迭代自我对弈次数
    checkpoint_freq = 1  # 保存间隔
    checkpoint_dir = "../../model/checkpoint"
    load_model = None  # 预训练模型路径


if __name__ == "__main__":
    trainer = Trainer(Args())
    trainer.learn()
