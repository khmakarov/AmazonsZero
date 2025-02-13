import torch
import torch.nn as nn


class AlphaZeroTrainer:

    def __init__(self, model, mcts, buffer_size=1000000):
        self.model = model
        self.mcts = mcts
        self.buffer = ReplayBuffer(buffer_size)

    def self_play(self, num_games=1000):
        for _ in range(num_games):
            game = GameCore()
            while not game.is_terminal():
                # 获取MCTS策略
                probs = self.mcts.get_action_probs(game.state())

                # 保存训练数据
                self.buffer.add(
                    state=game.state(),
                    policy=probs,
                    value=None  # 终局时填充
                )

                # 采样动作
                action = self._sample_action(probs)
                game.step(action)

            # 回填奖励值
            result = game.get_result()
            self.buffer.update_rewards(result)

    def train_epoch(self, batch_size=2048):
        states, policies, values = self.buffer.sample(batch_size)

        # 转换为张量
        states = torch.stack([preprocess(s) for s in states])
        policies = torch.tensor(policies)
        values = torch.tensor(values)

        # 前向传播
        pred_policies, pred_values = self.model(states)

        # 计算损失
        policy_loss = F.kl_div(pred_policies, policies)
        value_loss = F.mse_loss(pred_values, values)
        total_loss = policy_loss + value_loss

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return total_loss.item()
