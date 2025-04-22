import math
import numpy as np
import torch
from core.cpp.build.Amazons import GameCore


class MCTS():

    def __init__(self, nnet, cfg):
        self.nnet = nnet
        self.cpuct = cfg.cpuct
        self.num_simulations = cfg.num_simulations
        self.total_actions = cfg.TOTAL_ACTIONS

        self.Nsa = {}  # 存储每个状态的访问计数数组
        self.Psa = {}  # 存储策略概率数组
        self.Qsa = {}  # 存储每个状态的Q值数组
        self.Ns = {}  # 存储状态的访问次数
        self.Vs = {}  # 存储有效动作掩码

    def getActionProb(self, game, temp=1):
        try:
            s = game.compute_state_hash()
            valids_idx = game.get_legal_actions_np()
            s_hashes, s_states, s_valids = game.get_child_state_np(valids_idx)
            pis, _ = self.nnet.predict_batch(s_states, s_valids)
            for i in range(valids_idx[0]):
                child_hash = s_hashes[i]
                if child_hash not in self.Psa:
                    valid_actions = s_valids[i][1:s_valids[i][0] + 1]
                    self.Psa[child_hash] = pis[i][valid_actions] / np.sum(pis[i][valid_actions])  # 归一化
                    self.Qsa[child_hash] = np.zeros(s_valids[i][0], dtype=np.float32)
                    self.Nsa[child_hash] = np.zeros(s_valids[i][0], dtype=np.int32)
                    self.Vs[child_hash] = valid_actions
                    self.Ns[child_hash] = 1
            del pis, _
            for _ in range(self.num_simulations):
                self.search(game)
            counts = np.zeros(self.total_actions, dtype=np.int32)
            counts[self.Vs[s]] = self.Nsa[s]
            probs = np.power(counts, 1.0 / temp)
            probs = probs / np.sum(probs)
            return probs, valids_idx
        finally:
            torch.cuda.empty_cache()
            self.clear()

    def search(self, game):
        try:
            s = game.compute_state_hash()
            if s not in self.Psa:
                valids_idx = game.get_legal_actions_np()
                pi, v = self.nnet.predict(game.get_state_np(), valids_idx)
                valid_actions = valids_idx[1:valids_idx[0] + 1]
                self.Psa[s] = pi[valid_actions] / np.sum(pi[valid_actions])
                self.Qsa[s] = np.zeros(valids_idx[0], dtype=np.float32)
                self.Nsa[s] = np.zeros(valids_idx[0], dtype=np.int32)
                self.Vs[s] = valid_actions
                self.Ns[s] = 1
                return v * (1 - 2 * game.current_player)
            if self.Vs[s].size != 0:
                valid_actions = self.Vs[s]
                sqrt_Ns = math.sqrt(self.Ns[s])
                u_values = self.Qsa[s] + self.cpuct * self.Psa[s] * sqrt_Ns / (1 + self.Nsa[s])
                best_index = np.argmax(u_values)
                best_act = valid_actions[best_index]
                next_s = GameCore(game)
                next_s.step(best_act)
                v = self.search(next_s)
                self.Qsa[s][best_index] = (self.Nsa[s][best_index] * self.Qsa[s][best_index] + v) / (self.Nsa[s][best_index] + 1)
                self.Nsa[s][best_index] += 1
                self.Ns[s] += 1
                return -v
            else:
                return 1.0 if game.current_player == 1 else -1.0
        finally:
            torch.cuda.empty_cache()

    def clear(self):
        self.Nsa.clear()
        self.Psa.clear()
        self.Qsa.clear()
        self.Ns.clear()
        self.Vs.clear()
