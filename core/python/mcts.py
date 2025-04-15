import math
import numpy as np
from core.cpp.build.Amazons import GameCore


class MCTS():

    def __init__(self, nnet, cfg):
        self.nnet = nnet
        self.cpuct = cfg.cpuct
        self.num_simulations = cfg.num_simulations
        self.total_actions = cfg.TOTAL_ACTIONS

        self.Qsa = {}  # 存储每个状态的Q值数组
        self.Nsa = {}  # 存储每个状态的访问计数数组
        self.Ns = {}  # 存储状态的访问次数
        self.Ps = {}  # 存储策略概率数组
        self.Es = {}  # 存储终止状态
        self.Vs = {}  # 存储有效动作掩码

    def getActionProb(self, game, temp=1):
        for _ in range(self.num_simulations):
            self.search(game)

        s = game.stringRepresentation()
        valids_idx = game.get_legal_actions_np()
        n = valids_idx[0]
        counts = np.zeros(self.total_actions, dtype=np.int32)
        if s in self.Nsa:
            valids = valids_idx[1:n + 1]
            counts[valids] = self.Nsa[s][valids]
        else:
            counts[valids_idx[1:n + 1]] = 0
        if temp == 0:
            probs = np.zeros_like(counts, dtype=np.float32)
            bestA = np.argmax(counts)
            probs[bestA] = 1.0
        else:
            probs = np.power(counts, 1.0 / temp)
            probs = probs / np.sum(probs)

        return probs, valids_idx

    def search(self, game):
        s = game.stringRepresentation()

        if s not in self.Es:
            self.Es[s] = game.is_terminal()
        if self.Es[s] != 0:
            return -self.Es[s]
        if s not in self.Ps:
            valids_idx = game.get_legal_actions_np()
            valids = np.zeros(self.total_actions, dtype=np.bool_)
            valids[valids_idx[1:valids_idx[0] + 1]] = True
            self.Ps[s], v = self.nnet.predict(game, valids_idx)
            self.Ps[s] *= valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Qsa[s] = np.zeros(self.total_actions, dtype=np.float32)
            self.Nsa[s] = np.zeros(self.total_actions, dtype=np.int32)
            self.Vs[s] = np.array(valids_idx[1:valids_idx[0] + 1], dtype=np.int32)
            self.Ns[s] = 0
            return -v

        valid_actions = self.Vs[s]
        qs = self.Qsa[s][valid_actions]
        ps = self.Ps[s][valid_actions]
        ns = self.Nsa[s][valid_actions]

        sqrt_Ns = math.sqrt(self.Ns[s])
        u_values = qs + self.cpuct * ps * sqrt_Ns / (1 + ns)
        best_index = np.argmax(u_values)
        best_act = valid_actions[best_index]

        next_s = GameCore(game)
        next_s.step(best_act)
        v = self.search(next_s)

        self.Qsa[s][best_act] = (self.Nsa[s][best_act] * self.Qsa[s][best_act] + v) / (self.Nsa[s][best_act] + 1)
        self.Nsa[s][best_act] += 1
        self.Ns[s] += 1

        return -v
