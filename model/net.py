import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):
    """AlphaZero双头网络架构（策略头+价值头）"""

    def __init__(self, num_res_blocks=20, action_size=33344):
        super().__init__()

        # ----------------- 主干网络 -----------------
        # 输入层：5通道棋盘 -> 256通道
        self.input_conv = nn.Conv2d(
            in_channels=5,  # 输入通道：黑棋、白棋、障碍、当前玩家为黑则置1否则置0、当前玩家为白则置1否则置0
            out_channels=256,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.input_bn = nn.BatchNorm2d(256)

        # 残差块堆叠
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])

        # ----------------- 策略头 -----------------
        # 策略卷积：256通道 -> 2通道
        self.policy_conv = nn.Conv2d(
            in_channels=256,
            out_channels=2,  # 输出移动起始点和方向信息
            kernel_size=1,  # 1x1卷积融合通道信息
            bias=False
        )
        self.policy_bn = nn.BatchNorm2d(2)

        # 策略全连接：2x8x8 -> 动作空间
        self.policy_fc = nn.Linear(in_features=2 * 8 * 8, out_features=action_size)

        # ----------------- 价值头 -----------------
        # 价值卷积：256通道 -> 1通道
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)

        # 价值全连接
        self.value_fc1 = nn.Linear(1 * 8 * 8, 256)  # 展平后处理
        self.value_fc2 = nn.Linear(256, 1)  # 最终标量输出

    def forward(self, x):
        """
        前向传播
        输入形状： (batch_size, 5, 8, 8)
        输出：
            policy: (batch_size, action_size) 动作对数概率
            value:  (batch_size, 1) 当前玩家胜率估计 [-1, 1]
        """
        # ----------------- 主干网络 -----------------
        x = self.input_conv(x)
        x = self.input_bn(x)
        x = F.relu(x, inplace=True)
        x = self.res_blocks(x)  # 通过残差块

        # ----------------- 策略头 -----------------
        p = self.policy_conv(x)
        p = self.policy_bn(p)
        p = F.relu(p, inplace=True)
        p = p.view(p.size(0), -1)  # 展平：(batch, 2*8*8)
        p = self.policy_fc(p)
        policy = F.log_softmax(p, dim=1)  # 对数概率

        # ----------------- 价值头 -----------------
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)  # 展平：(batch, 1*8*8)
        v = F.relu(self.value_fc1(v))
        v = self.value_fc2(v)
        value = torch.tanh(v)  # 压缩到[-1, 1]

        return policy, value
