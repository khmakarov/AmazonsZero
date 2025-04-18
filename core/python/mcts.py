import math
import numpy as np
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
        game_ = GameCore(game)
        for _ in range(self.num_simulations):
            self.search(game_)

        s = game.compute_state_hash()
        valids_idx = game.get_legal_actions_np()
        counts = np.zeros(self.total_actions, dtype=np.int32)

        if s in self.Nsa:
            counts[self.Vs[s]] = self.Nsa[s][self.Vs[s]]
        else:
            counts[valids_idx[1:valids_idx[0] + 1]] = 0

        if temp == 0:
            probs = np.zeros_like(counts, dtype=np.float32)
            bestA = np.argmax(counts)
            probs[bestA] = 1.0
        else:
            probs = np.power(counts, 1.0 / temp)
            probs = probs / np.sum(probs)

        self.clear()
        return probs, valids_idx

    def search(self, game):
        s = game.compute_state_hash()

        if s not in self.Psa:
            valids_idx = game.get_legal_actions_np()
            valids = np.zeros(self.total_actions, dtype=np.bool_)
            valids[valids_idx[1:valids_idx[0] + 1]] = True
            self.Psa[s], v = self.nnet.predict(game, valids_idx)
            sum_Psa_s = np.sum(self.Psa[s])
            if sum_Psa_s > 0:
                self.Psa[s] /= sum_Psa_s
            else:
                self.Psa[s] = self.Psa[s] + valids
                self.Psa[s] /= np.sum(self.Psa[s])

            self.Qsa[s] = np.zeros(self.total_actions, dtype=np.float32)
            self.Nsa[s] = np.zeros(self.total_actions, dtype=np.int32)
            self.Vs[s] = valids_idx[1:valids_idx[0] + 1]
            self.Ns[s] = 1
            return v * (1 - 2 * game.current_player)
        if self.Vs[s].size != 0:
            valid_actions = self.Vs[s]
            qsa = self.Qsa[s][valid_actions]
            psa = self.Psa[s][valid_actions]
            nsa = self.Nsa[s][valid_actions]

            sqrt_Ns = math.sqrt(self.Ns[s])
            u_values = qsa + self.cpuct * psa * sqrt_Ns / (1 + nsa)
            best_index = np.argmax(u_values)
            best_act = valid_actions[best_index]

            next_s = GameCore(game)
            next_s.step(best_act)
            v = self.search(next_s)

            self.Qsa[s][best_act] = (self.Nsa[s][best_act] * self.Qsa[s][best_act] + v) / (self.Nsa[s][best_act] + 1)
            self.Nsa[s][best_act] += 1
            self.Ns[s] += 1

            return -v
        else:
            return 1.0 if game.current_player == 1 else -1.0

    def clear(self):
        self.Nsa.clear()
        self.Psa.clear()
        self.Qsa.clear()
        self.Ns.clear()
        self.Vs.clear()
