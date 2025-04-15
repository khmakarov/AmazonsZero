import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 16, kernel_size=1), nn.ReLU(),
            nn.Conv2d(channels // 16, channels, kernel_size=1), nn.Sigmoid()
        )

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x * self.se(x)
        x += residual
        return F.relu(x)


class AlphaZeroNet(nn.Module):

    def __init__(self, num_res_blocks=30, action_size=33344):
        super().__init__()

        self.input_conv = nn.Conv2d(5, 256, kernel_size=3, padding=1, bias=False)  # <- 修改处
        self.input_bn = nn.BatchNorm2d(256)

        # 残差块
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])

        # 策略头
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Sequential(nn.Linear(2 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, action_size))

        # 价值头
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc = nn.Sequential(nn.Linear(1 * 8 * 8, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, x, valids_idx=None):
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
        p = p.reshape(p.size(0), -1)  # 展平：(batch, 2*8*8)
        p = self.policy_fc(p)

        if valids_idx is not None:
            batch_size = x.size(0)
            max_length = valids_idx[:, 0].max().item()

            indices = valids_idx[:, 1:max_length + 1]  # (batch_size, max_length)
            batch_idx = torch.arange(batch_size, device=x.device)[:, None].expand(-1, max_length)
            valid_mask = (indices != -1)
            valids = torch.zeros((batch_size, 33344), dtype=torch.bool, device=x.device)
            valids[(batch_idx[valid_mask], indices[valid_mask])] = True

            p = p.masked_fill(~valids, -1e9)

        policy = F.log_softmax(p, dim=1)  # 对数概率

        # ----------------- 价值头 -----------------
        v = self.value_conv(x)
        v = self.value_bn(v)
        v = F.relu(v, inplace=True)
        v = v.view(v.size(0), -1)  # 展平：(batch, 1*8*8)
        v = self.value_fc(v)
        value = torch.tanh(v)  # 压缩到[-1, 1]

        return policy, value

    def predict(self, state, valids_idx):
        self.eval()
        with torch.no_grad():
            state = state.get_state_np()
            state = torch.tensor(state, dtype=torch.float32)
            state = state.permute(2, 0, 1).unsqueeze(0).to('cuda')  # HWC -> NCHW
            valids_idx = torch.as_tensor(valids_idx, device='cuda').unsqueeze(0)
            log_pi, v = self(state, valids_idx)
            return torch.exp(log_pi).cpu().numpy()[0], v.cpu().numpy()[0][0]
