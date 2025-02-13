# model/resnet.py
import torch
import torch.nn as nn


class AmazonsResNet(nn.Module):

    def __init__(self, in_channels=5, res_blocks=20):
        super().__init__()

        # 输入层
        self.conv_in = nn.Conv2d(in_channels, 256, 3, padding=1)
        self.bn_in = nn.BatchNorm2d(256)

        # 残差块
        self.res_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(256, 256,
                                    3, padding=1), nn.BatchNorm2d(256),
                          nn.ReLU(), nn.Conv2d(256, 256, 3, padding=1),
                          nn.BatchNorm2d(256)) for _ in range(res_blocks)
        ])

        # 策略头
        self.policy_conv = nn.Conv2d(256, 2, 1)
        self.policy_fc = nn.Linear(8 * 8 * 2, 4096)

        # 价值头
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_fc = nn.Sequential(nn.Linear(8 * 8, 256), nn.ReLU(),
                                      nn.Linear(256, 1), nn.Tanh())

    def forward(self, x):
        # 输入形状: (batch, 5, 8, 8)
        x = torch.relu(self.bn_in(self.conv_in(x)))

        # 残差连接
        for block in self.res_blocks:
            residual = x
            x = block(x)
            x += residual
            x = torch.relu(x)

        # 策略输出
        policy = self.policy_conv(x)
        policy = self.policy_fc(policy.view(-1, 8 * 8 * 2))

        # 价值输出
        value = self.value_conv(x)
        value = self.value_fc(value.view(-1, 8 * 8))

        return policy, value
