# evaluate.py
from unittest import result
import numpy as np
import torch
import torch.multiprocessing as mp
from core.cpp.build.Amazons import GameCore
from core.base.build.Amazons import Game
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from src.utils.data_manager import DataManager


class Evaluator:

    def __init__(self, cfg):
        self.data_mgr = DataManager()
        self.mcts_config = cfg.mcts
        self.iteration = 0
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

        with mp.Pool(processes=16) as pool:
            tasks = [(self.mcts_config, model_a, model_b) for _ in range(self.num_games)]
            results = pool.starmap(self.evaluate_game_worker_mcts, tasks)
            pool.close()
            pool.join()
        self._process_episodes(results)
        self._cleanup_shared_models()
        self.data_mgr.flush_visual_data()
        a_wins = [result[2] for result in results]
        return (sum(a_wins) + 1) / (self.num_games + 2)

    def evaluate_with_cpp(self, iteration, model_a_path):
        print("开始评估")
        self.iteration = iteration

        if model_a_path not in self.shared_models:
            self.shared_models[model_a_path] = self._load_shared_model(model_a_path)
        model_a = self.shared_models[model_a_path]
        with mp.Pool(processes=1) as pool:
            tasks = [(self.mcts_config, model_a) for _ in range(self.num_games)]
            results = pool.starmap(self.evaluate_game_worker1, tasks)
            pool.close()
            pool.join()
        self._process_episodes(results)
        self._cleanup_shared_models()
        self.data_mgr.flush_visual_data()
        a_wins = [result[2] for result in results]
        return (sum(a_wins) + 1) / (self.num_games + 2)

    def _load_shared_model(self, path):
        """加载模型并通过共享内存共享"""
        model = AlphaZeroNet().to("cuda")
        checkpoint = torch.load(path, weights_only=False)
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
    def evaluate_game_worker(model_a, model_b):
        try:
            current_model = np.random.choice([model_a, model_b])
            episode_data = []
            start = 0 if current_model == model_a else 1
            state = GameCore()
            while True:
                ended = state.is_terminal()
                if ended == 0:
                    valids_idx = state.get_legal_actions_np()
                    pi, _ = current_model.predict(state.get_state_np(), valids_idx)
                    n = min(10, len(pi))  # 实际有效走法数
                    sorted_indices = np.argsort(pi)[::-1]  # 降序排列
                    valid_indices = sorted_indices[:n]  # 前n个有效索引
                    valid_probs = pi[valid_indices].tolist()
                    top5_data = list(zip(valid_indices.tolist(), valid_probs))
                    action_index = np.argmax(pi)
                    next_state = GameCore(state)
                    next_state.step(action_index)
                    episode_data.append([state.get_state_np(), state.index2action(action_index), top5_data])
                    state = next_state
                    current_model = model_b if current_model == model_a else model_a
                else:
                    episode_data.append([state.get_state_np(), state.index2action(0), []])
                    a_win = 1 if (start == 0 and ended == 1) or (start == 1 and ended == -1) else 0
                    break
            return (episode_data, ended, a_win)
        finally:
            del current_model
            torch.cuda.empty_cache()

    @staticmethod
    def evaluate_game_worker_mcts(mcts_config, model_a, model_b):
        try:
            current_model = np.random.choice([model_a, model_b])
            episode_data = []
            mcts = MCTS(model_a, mcts_config)
            start = 0 if current_model == model_a else 1
            state = GameCore()
            while True:
                ended = state.is_terminal()
                if ended == 0:
                    pi, _, _ = mcts.getActionProb(state)
                    n = min(10, len(pi))  # 实际有效走法数
                    sorted_indices = np.argsort(pi)[::-1]  # 降序排列
                    valid_indices = sorted_indices[:n]  # 前n个有效索引
                    valid_probs = pi[valid_indices].tolist()
                    top5_data = list(zip(valid_indices.tolist(), valid_probs))
                    action_index = np.argmax(pi)
                    next_state = GameCore(state)
                    next_state.step(action_index)
                    episode_data.append([state.get_state_np(), state.index2action(action_index), top5_data])
                    state = next_state
                    current_model = model_b if current_model == model_a else model_a
                else:
                    episode_data.append([state.get_state_np(), state.index2action(0), []])
                    a_win = 1 if (start == 0 and ended == 1) or (start == 1 and ended == -1) else 0
                    break
            return (episode_data, ended, a_win)
        finally:
            del current_model
            torch.cuda.empty_cache()

    @staticmethod
    def evaluate_game_worker1(mcts_config, model):
        try:
            torch.cuda.init()  # 初始化 CUDA
            stream = torch.cuda.Stream()  # 创建独立的 CUDA 流
            with torch.cuda.stream(stream):
                torch.cuda.set_device(0)  # 指定 GPU 设备
                model.to("cuda")  # 强制模型加载到 GPU
                episode_data = []
                state = GameCore()
                action_idx = np.zeros(56, dtype=np.int32)
                turnID = 0
                print(f"[子进程] CUDA 可用: {torch.cuda.is_available()}")
                print(f"[子进程] 当前设备: {torch.cuda.current_device()}")
                while True:
                    ended = state.is_terminal()
                    game = Game()
                    if ended == 0:
                        if turnID % 2 == 0:
                            valids_idx = state.get_legal_actions_np()
                            pi, _ = model.predict(state.get_state_np(), valids_idx)
                            n = min(10, len(pi))  # 实际有效走法数
                            sorted_indices = np.argsort(pi)[::-1]  # 降序排列
                            valid_indices = sorted_indices[:n]  # 前n个有效索引
                            valid_probs = pi[valid_indices].tolist()
                            top5_data = list(zip(valid_indices.tolist(), valid_probs))
                            action_index = np.argmax(pi)
                        else:
                            actions, probs = game.step(turnID, action_idx)
                            actions = np.array(actions)
                            probs = np.array(probs)
                            action_index = actions[0]
                            top5_data = list(zip(actions.tolist(), probs.tolist()))
                        next_state = GameCore(state)
                        next_state.step(action_index)
                        episode_data.append([state.get_state_np(), state.index2action(action_index), top5_data])
                        action_idx[turnID] = action_index
                        turnID += 1
                        state = next_state
                        del game
                    else:
                        episode_data.append([state.get_state_np(), state.index2action(0), []])
                        a_win = 1 if ended == 1 else 0
                        break
                return (episode_data, ended, a_win)
        finally:
            torch.cuda.empty_cache()
