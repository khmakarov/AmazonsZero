# train.py
import torch
import torch.optim as optim
import numpy as np
from collections import deque
import random
from mcts import MCTS
from net import AlphaZeroNet
from amazons_core import GameCore
import os

TOTAL_ACTIONS = 33344


class Trainer:

    def __init__(self, args):
        self.args = args
        self.game = self._init_game()
        self.nnet = self._init_net()
        self.optimizer = optim.Adam(self.nnet.parameters(), lr=args.lr)
        self.mcts = MCTS(self.game, self.nnet, args)
        self.train_examples = []
        self.batch_size = args.batch_size
        self.num_iters = args.num_iters
        self.num_eps = args.num_eps
        self.checkpoint_dir = args.checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def _init_game(self):

        class AmazonsGameWrapper:

            def __init__(self):
                self.core = GameCore()
                self.action_size = TOTAL_ACTIONS

            def getGameEnded(self, state, player):
                terminal, winner = state.is_terminal()
                if not terminal:
                    return 0
                return 1 if winner == player else -1

            def getValidMoves(self, state):
                return np.array(state.get_legal_actions(), dtype=np.float32)

            def getNextState(self, state, action_index):
                new_state = state.clone()
                new_state.step(action_index)
                return new_state, new_state.current_player

            def getActionSize(self):
                return self.action_size

            def stringRepresentation(self, state):
                return state.stringRepresentation()

        return AmazonsGameWrapper()

    def _init_net(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nnet = AlphaZeroNet(action_size=TOTAL_ACTIONS).to(device)
        if self.args.load_model:
            checkpoint = torch.load(self.args.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def execute_episode(self):
        examples = []
        state = self.game.core.clone()
        while True:
            canonical_state = state
            temp = 1 if len(examples) < 30 else 0.1  # 前30步探索，后续利用
            pi = self.mcts.getActionProb(canonical_state, temp=temp)
            examples.append([canonical_state, pi, None])

            action = np.random.choice(len(pi), p=pi)
            next_state, _ = self.game.getNextState(state, action)
            ended = self.game.getGameEnded(next_state, 1)

            if ended != 0:
                return [(x[0], x[1], ended) for x in examples]
            state = next_state

    def learn(self):
        for i in range(1, self.num_iters + 1):
            print(f"Iteration {i}")
            # 自我对弈生成数据
            eps_data = []
            for _ in range(self.num_eps):
                eps_data += self.execute_episode()
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


def main():

    class Args:
        numMCTSSims = 128  # MCTS模拟次数
        cpuct = 1.0  # 探索系数
        lr = 0.001  # 学习率
        batch_size = 2048  # 训练批次大小
        num_iters = 1000  # 训练总迭代次数
        num_eps = 100  # 每迭代自我对弈次数
        checkpoint_freq = 50  # 保存间隔
        checkpoint_dir = "model/checkpoints"
        load_model = None  # 预训练模型路径

    trainer = Trainer(Args())
    trainer.learn()


if __name__ == "__main__":
    main()
