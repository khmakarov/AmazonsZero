import logging
import math
import numpy as np
from Amazons import GameCore

EPS = 1e-8
TOTAL_ACTIONS = 33344

log = logging.getLogger(__name__)


class MCTS():

    def __init__(self, nnet, args):
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
        self.Nsa = {}  # stores #times edge s,a was visited
        self.Ns = {}  # stores #times board s was visited
        self.Ps = {}  # stores initial policy (returned by neural net)

        self.Es = {}  # stores game.getGameEnded ended for board s
        self.Vs = {}  # stores game.getValidMoves for board s

    def getActionProb(self, game, temp=1):

        for i in range(self.args.numMCTSSims):
            self.search(game)

        s = game.stringRepresentation()
        _, valids_idx = game.get_legal_actions()
        n = valids_idx[0]
        counts = np.zeros(TOTAL_ACTIONS, dtype=np.int32)
        valids_idx = np.array(valids_idx[1:n + 1], dtype=np.int32)
        counts[valids_idx] = np.fromiter((self.Nsa.get((s, a), 0) for a in valids_idx), dtype=np.int32, count=len(valids_idx))

        if temp == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x**(1. / temp) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs

    def search(self, game):

        s = game.stringRepresentation()
        valids, valids_idx = game.get_legal_actions()
        valids = np.array(valids)
        if s not in self.Es:
            self.Es[s] = game.is_terminal()
        if self.Es[s] != 0:
            return -self.Es[s]
        if s not in self.Ps:
            self.Ps[s], v = self.nnet.predict(np.array(game.get_state()))
            self.Ps[s] = self.Ps[s] * valids
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s  # renormalize
            else:
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_act = -1
        n = valids_idx[0]
        for a in valids_idx[1:n + 1]:
            if (s, a) in self.Qsa:
                u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            else:
                u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?

            if u > cur_best:
                cur_best = u
                best_act = a

        a = best_act
        next_s = GameCore(game)
        next_s.step(a)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1

        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1

        self.Ns[s] += 1
        return -v
