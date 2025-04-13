# evaluate.py
import os
import numpy as np
import torch
import torch.multiprocessing as mp
from core.cpp.build.Amazons import GameCore
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet


class Evaluator:

    def __init__(self, cfg):
        self.mcts = cfg.mcts
        self.num_games = cfg.evaluator.num_games
        self.shared_models = {}

    def evaluate(self, model_a_path, model_b_path):
        print("开始评估")
        if model_a_path not in self.shared_models:
            self.shared_models[model_a_path] = self._load_shared_model(model_a_path)
        if model_b_path not in self.shared_models:
            self.shared_models[model_b_path] = self._load_shared_model(model_b_path)
        model_a = self.shared_models[model_a_path]
        model_b = self.shared_models[model_b_path]

        # 显式管理进程池生命周期
        pool = mp.Pool(processes=os.cpu_count())
        tasks = [(model_a, model_b, self.mcts) for _ in range(self.num_games)]
        results = pool.starmap(self.evaluate_game_worker, tasks)
        pool.close()
        pool.join()  # 等待所有子进程退出

        # 所有子进程结束后再清理模型
        self._cleanup_shared_models()
        return sum(results) / self.num_games

    def _load_shared_model(self, path):
        """加载模型并通过共享内存共享"""
        model = AlphaZeroNet().to("cuda")
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        model.share_memory()  # 启用跨进程共享
        return model.eval()

    def _cleanup_shared_models(self):
        """显式解除共享内存绑定（确保所有子进程已退出）"""
        self.shared_models.clear()
        torch.cuda.empty_cache()

    @staticmethod
    def evaluate_game_worker(model_a, model_b, cfg):
        try:
            with torch.cuda.device("cuda"):
                mcts_a = MCTS(model_a, cfg)
                mcts_b = MCTS(model_b, cfg)
                current_model = np.random.choice([model_a, model_b])
                start = 0 if current_model == model_a else 1
                state = GameCore()

                while True:
                    mcts = mcts_a if current_model == model_a else mcts_b
                    pi, _ = mcts.getActionProb(state)
                    action_index = np.argmax(pi)
                    next_state = GameCore(state)
                    next_state.step(action_index)
                    ended = next_state.is_terminal()

                    if ended != 0:
                        winner = "黑胜" if (ended == 1 and next_state.current_player == 0) else "白胜"
                        a_win = (start == 0 and winner == "黑胜") or (start == 1 and winner == "白胜")
                        break
                    else:
                        state = next_state
                        current_model = model_b if current_model == model_a else model_a

                return 1 if a_win else 0
        finally:
            # 显式释放子进程对共享模型的引用
            del mcts_a, mcts_b, current_model, state
            torch.cuda.empty_cache()
