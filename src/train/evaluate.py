# evaluate.py
import numpy as np
import torch
import torch.multiprocessing as mp
from core.cpp.build.Amazons import GameCore
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from src.utils.data_manager import DataManager


class Evaluator:

    def __init__(self, cfg):
        self.data_mgr = DataManager()
        self.iteration = 0
        self.mcts = cfg.mcts
        self.num_games = cfg.evaluator.num_games
        self.shared_models = {}

    def evaluate(self, iteration, model_a_path, model_b_path):
        print("开始评估")
        self.iteration = iteration

        if model_a_path not in self.shared_models:
            self.shared_models[model_a_path] = self._load_shared_model(model_a_path)
        if model_b_path not in self.shared_models:
            self.shared_models[model_b_path] = self._load_shared_model(model_b_path)
        model_a = self.shared_models[model_a_path]
        model_b = self.shared_models[model_b_path]

        with mp.Pool(processes=8) as pool:
            tasks = [(model_a, model_b, self.mcts) for _ in range(self.num_games)]
            results = pool.starmap(self.evaluate_game_worker, tasks)
            pool.close()
            pool.join()
        self._process_episodes(results)
        self._cleanup_shared_models()
        self.data_mgr.flush_visual_data()
        a_wins = [result[2] for result in results]
        return sum(a_wins) / self.num_games

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

    def _process_episodes(self, results):
        print("评估数据生成完毕")
        for visual_data, ended, _ in results:
            self.data_mgr.add_visual_data(visual_data, ended, self.iteration, 1)

    @staticmethod
    def evaluate_game_worker(model_a, model_b, cfg):
        try:
            current_model = np.random.choice([model_a, model_b])
            episode_data = []
            mcts_a = MCTS(model_a, cfg)
            mcts_b = MCTS(model_b, cfg)
            start = 0 if current_model == model_a else 1
            state = GameCore()
            while True:
                ended = state.is_terminal()
                if ended == 0:
                    mcts = mcts_a if current_model == model_a else mcts_b
                    pi, _ = mcts.getActionProb(state)
                    #action_index = np.argmax(pi)
                    action_index = np.random.choice(len(pi), p=pi)
                    next_state = GameCore(state)
                    next_state.step(action_index)
                    episode_data.append([state.get_state_np(), state.index2action(action_index)])
                    state = next_state
                    current_model = model_b if current_model == model_a else model_a
                else:
                    print(" 对局结束")
                    episode_data.append([state.get_state_np(), state.index2action(0)])
                    a_win = 1 if (start == 0 and ended == 1) or (start == 1 and ended == -1) else 0
                    break

            return (episode_data, ended, a_win)
        finally:
            print(" 子进程返回")
            del mcts_a, mcts_b, current_model
            torch.cuda.empty_cache()
