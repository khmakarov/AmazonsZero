import math
import random
import numpy as np
import torch
from core.cpp.build.Amazons import GameCore


def smooth_top_values(pi_cut, tau=1.0):
    original_sum = np.sum(pi_cut)
    scaled = pi_cut / tau
    scaled -= np.max(scaled)
    exp_values = np.exp(scaled)
    softmax = exp_values / np.sum(exp_values)
    smoothed = softmax * original_sum
    return smoothed


def add_dirichlet_noise(prior, alpha=0.03, epsilon=0.25):
    noise = np.random.dirichlet([alpha] * len(prior))
    return (1 - epsilon) * prior + epsilon * noise


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
            for _ in range(min(random.randint(100, 200), valids_idx[0] * 2)):
                self.search(game)
            counts = np.zeros(self.total_actions, dtype=np.int16)
            counts[self.Vs[s]] = self.Nsa[s]

            probs = np.power(counts, 1.0 / temp)
            probs = probs / np.sum(probs)
            sum_n = np.sum(self.Nsa[s])
            v = np.sum(self.Nsa[s] * self.Qsa[s]) / sum_n
            return probs, valids_idx, v
        finally:
            torch.cuda.empty_cache()
            self.clear()

    def _initialize_node(self, state_hash, policy, valid_actions):
        policy_valid = policy[valid_actions]
        policy_sum = policy_valid.sum()
        policy_normalized = policy_valid / policy_sum

        num_actions = min(12, len(policy_normalized))
        partitioned_indices = np.argpartition(-policy_normalized, num_actions - 1)[:num_actions]
        sorted_sub_indices = np.argsort(-policy_normalized[partitioned_indices])
        cut_indices = partitioned_indices[sorted_sub_indices]
        cut = valid_actions[cut_indices]

        self.Psa[state_hash] = smooth_top_values(add_dirichlet_noise(policy[cut]))
        self.Vs[state_hash] = cut
        self.Qsa[state_hash] = np.zeros(len(cut), dtype=np.float32)
        self.Nsa[state_hash] = np.zeros(len(cut), dtype=np.int32)
        self.Ns[state_hash] = 1

    def search(self, game):
        s = game.compute_state_hash()
        if s not in self.Psa:
            valids_idx = game.get_legal_actions_np()
            valid_actions = valids_idx[1:valids_idx[0] + 1]
            pi, v = self.nnet.predict(game.get_state_np(), valids_idx)
            if valid_actions.size != 0:
                self._initialize_node(s, pi, valid_actions)
                cut = self.Vs[s]
                s_hashes, s_states, s_valids = game.get_child_state_np(cut)
                pis, vs = self.nnet.predict_batch(s_states, s_valids)
                self.Qsa[s] = -vs.astype(np.float32)
                self.Nsa[s] = np.ones(len(cut), dtype=np.int32)
                self.Ns[s] = len(cut)
                for i in range(len(cut)):
                    child_hash = int(s_hashes[i])
                    if child_hash not in self.Psa:
                        valid_actions = s_valids[i][1:s_valids[i][0] + 1]
                        if valid_actions.size != 0:
                            self._initialize_node(child_hash, pis[i], valid_actions)
                return -v, vs
            else:
                return 1.0, np.zeros(2, dtype=np.int16)
        if self.Vs[s].size != 0:
            valid_actions = self.Vs[s]
            sqrt_Ns = math.sqrt(self.Ns[s])
            u_values = self.Qsa[s] + self.cpuct * self.Psa[s] * sqrt_Ns / (1 + self.Nsa[s])
            best_index = np.argmax(u_values)
            best_act = valid_actions[best_index]
            next_s = GameCore(game)
            next_s.step(best_act)
            v, vs = self.search(next_s)
            self.Qsa[s][best_index] = (self.Nsa[s][best_index] * self.Qsa[s][best_index] + v) / (self.Nsa[s][best_index] + 1)
            self.Nsa[s][best_index] += 1
            self.Ns[s] += 1
            if np.any(vs != 0):
                sum_vs = vs.sum()
                k = vs.size
                original_Nsa = self.Nsa[s][best_index]
                self.Qsa[s][best_index] = (original_Nsa * self.Qsa[s][best_index] + sum_vs) / (original_Nsa + k)
                self.Nsa[s][best_index] += k
                self.Ns[s] += k
            return -v, -vs
        else:
            return 1.0, np.zeros(2, dtype=np.int16)

    def clear(self):
        self.Nsa.clear()
        self.Psa.clear()
        self.Qsa.clear()
        self.Ns.clear()
        self.Vs.clear()
