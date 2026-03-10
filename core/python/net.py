import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.bn1 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01)

        # 第二个卷积层（修正初始化）
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.conv2.weight, mode='fan_out', nonlinearity='relu')
        self.bn2 = nn.BatchNorm2d(channels, eps=1e-5, momentum=0.01)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x, inplace=True)
        return x


class ValidMask(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, p, valids_idx):
        batch_size = p.size(0)
        k = valids_idx[:, 0].long()
        max_k = k.max().item()

        indices = valids_idx[:, 1:1 + max_k]
        row_indices = torch.arange(batch_size, device=p.device).unsqueeze(1).expand(-1, max_k)
        mask = torch.arange(max_k, device=p.device).expand(batch_size, max_k) < k.unsqueeze(1)

        valid_rows = row_indices[mask]
        valid_cols = indices[mask]

        valids = torch.zeros((batch_size, 33344), dtype=torch.bool, device=p.device)
        valids[valid_rows, valid_cols] = True

        return p.masked_fill(~valids, -1e4), valids


class AlphaZeroNet(nn.Module):

    def __init__(self, num_res_blocks=20, action_size=33344):
        super().__init__()
        # ----------------- 输入层初始化 -----------------
        self.input_conv = nn.Conv2d(5, 256, kernel_size=3, padding=1, bias=False)
        nn.init.kaiming_normal_(self.input_conv.weight, mode='fan_out', nonlinearity='relu')
        self.input_bn = nn.BatchNorm2d(256, eps=1e-5, momentum=0.01)

        # ----------------- 残差块初始化 -----------------
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(num_res_blocks)])

        # ----------------- 策略头初始化 -----------------
        self.policy_conv = nn.Conv2d(256, 32, kernel_size=1, bias=False)
        nn.init.kaiming_normal_(self.policy_conv.weight, mode='fan_out', nonlinearity='relu')
        self.policy_bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01)
        self.policy_fc = nn.Sequential(nn.Linear(32 * 8 * 8, 512), nn.ReLU(inplace=True), nn.LayerNorm(512), nn.Linear(512, action_size))
        nn.init.kaiming_normal_(self.policy_fc[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.normal_(self.policy_fc[-1].weight, mean=0, std=0.01)

        # ----------------- 价值头初始化 -----------------
        self.value_conv = nn.Conv2d(256, 32, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(32, eps=1e-5, momentum=0.01)
        self.value_fc = nn.Sequential(nn.Linear(32 * 8 * 8, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh())

        nn.init.kaiming_normal_(self.value_conv.weight, mode='fan_out', nonlinearity='relu')
        nn.init.zeros_(self.value_fc[2].weight)
        nn.init.zeros_(self.value_fc[2].bias)
        self.valid_mask = ValidMask()

    def forward(self, x, valids_idx=None):
        x = F.relu(self.input_bn(self.input_conv(x)), inplace=True)
        x = self.res_blocks(x)

        p = self.policy_conv(x)
        p = F.relu(self.policy_bn(p), inplace=True)
        p = p.reshape(p.size(0), -1)
        p = self.policy_fc(p)
        p, valids = self.valid_mask(p, valids_idx)
        policy = F.log_softmax(p, dim=1)

        v = F.relu(self.value_bn(self.value_conv(x)), inplace=True)
        v = v.reshape(v.size(0), -1)
        value = self.value_fc(v)

        return policy, value, valids

    def predict(self, state, valids_idx):
        self.eval()
        with torch.no_grad():
            try:
                state = torch.tensor(state, dtype=torch.float32)
                state = state.permute(2, 0, 1).unsqueeze(0).to('cuda')
                valids_idx = torch.as_tensor(valids_idx, device='cuda').unsqueeze(0)
                with torch.amp.autocast("cuda"):
                    log_pi, v, _ = self(state, valids_idx)
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
                with torch.amp.autocast("cuda"):
                    log_pi, v, _ = self(states, valids_idx)
                pi = torch.exp(log_pi).cpu().numpy()
                v = v.cpu().numpy().squeeze(axis=1)
                return pi, v
            finally:
                del states, valids_idx, log_pi, v
