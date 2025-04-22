import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 第一个卷积层：He初始化 + BN
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

        # 第二个卷积层：零初始化 + BN（恒等映射初始化）
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.zeros_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-3, momentum=0.01)

        # SE模块：最后一层卷积初始化为零
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels // 16, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv2d(channels // 16, channels, kernel_size=1, bias=False), nn.Sigmoid()
        )
        nn.init.kaiming_normal_(self.se[1].weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.se[3].weight)  # SE最后一层初始化为零

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = x * self.se(x)
        x += residual
        return F.relu(x, inplace=True)


class AlphaZeroNet(nn.Module):

    def __init__(self, num_res_blocks=20, action_size=33344):
        super().__init__()
        # ----------------- 输入层 -----------------
        self.input_conv = nn.Conv2d(5, 256, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='relu')
        self.input_bn = nn.BatchNorm2d(256, eps=1e-3, momentum=0.01)

        # ----------------- 残差块 -----------------
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])

        # ----------------- 策略头 -----------------
        self.policy_conv = nn.Conv2d(256, 2, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.policy_conv.weight, mode='fan_out', nonlinearity='relu')
        self.policy_bn = nn.BatchNorm2d(2, eps=1e-3, momentum=0.01)
        self.policy_fc = nn.Sequential(
            nn.Linear(2 * 8 * 8, 512),
            nn.ReLU(inplace=True),
            nn.LayerNorm(512),  # 层标准化防止过拟合
            nn.Linear(512, action_size)
        )
        nn.init.xavier_normal_(self.policy_fc[0].weight)
        nn.init.xavier_normal_(self.policy_fc[-1].weight, gain=0.1)  # 缩小输出范围

        # ----------------- 价值头 -----------------
        self.value_conv = nn.Conv2d(256, 1, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        self.value_bn = nn.BatchNorm2d(1, eps=1e-3, momentum=0.01)
        self.value_fc = nn.Sequential(
            nn.Linear(1 * 8 * 8, 512), nn.ReLU(inplace=True), nn.Linear(512, 256), nn.ReLU(inplace=True), nn.Linear(256, 1)
        )
        # 输出层初始化为零，防止tanh过早饱和
        nn.init.zeros_(self.value_fc[-1].weight)
        nn.init.zeros_(self.value_fc[-1].bias)

    def forward(self, x, valids_idx=None):
        # ----------------- 主干网络 -----------------
        x = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        x = self.res_blocks(x)

        # ----------------- 策略头 -----------------
        p = self.policy_conv(x)
        p = F.relu(self.policy_bn(p), inplace=True)
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)

        # 掩码无效动作
        if valids_idx is not None:
            batch_size = x.size(0)
            max_length = valids_idx[:, 0].max().item()
            indices = valids_idx[:, 1:max_length + 1]
            batch_idx = torch.arange(batch_size, device=x.device)[:, None]
            valid_mask = (indices != -1)
            valids = torch.zeros((batch_size, 33344), dtype=torch.bool, device=x.device)
            valids[batch_idx.expand_as(indices)[valid_mask], indices[valid_mask]] = True
            p = p.masked_fill(~valids, -1e9)

        policy = F.log_softmax(p, dim=1)

        # ----------------- 价值头 -----------------
        v = self.value_conv(x)
        v = F.relu(self.value_bn(v), inplace=True)
        v = v.view(v.size(0), -1)
        v = self.value_fc(v)
        value = torch.tanh(v)

        return policy, value

    def predict(self, state, valids_idx):
        self.eval()
        with torch.no_grad():
            try:
                state = torch.tensor(state, dtype=torch.float32)
                state = state.permute(2, 0, 1).unsqueeze(0).to('cuda')
                valids_idx = torch.as_tensor(valids_idx, device='cuda').unsqueeze(0)
                log_pi, v = self(state, valids_idx)
                return torch.exp(log_pi).cpu().numpy()[0], v.cpu().numpy()[0][0]
            finally:
                del state, valids_idx, log_pi, v

    def predict_batch(self, states, valids_idx):
        self.eval()
        with torch.no_grad():
            try:
                states = torch.as_tensor(states, dtype=torch.float32)
                states = states.permute(0, 3, 1, 2).to('cuda')
                valids_idx = torch.as_tensor(valids_idx, dtype=torch.int, device='cuda')
                log_pi, v = self(states, valids_idx)
                pi = torch.exp(log_pi).cpu().numpy()
                v = v.cpu().numpy().squeeze(axis=1)
                return pi, v
            finally:
                del states, valids_idx, log_pi, v
