# evaluate.py
import os
import torch
import numpy as np
import multiprocessing
from datetime import datetime
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from Amazons import GameCore


class Evaluator:

    def __init__(self, cfg):
        self.mcts = cfg.mcts
        self.num_games = cfg.evaluator.num_games

    def evaluate_models(self, model_a_path, model_b_path):
        with multiprocessing.Pool(processes=os.cpu_count()) as pool:
            tasks = [(model_a_path, model_b_path, self.mcts) for _ in range(self.num_games)]
            results = pool.starmap(self.evaluate_game_worker, tasks)
        return sum(results) / self.num_games

    @staticmethod
    def evaluate_game_worker(model_a_path, model_b_path, cfg):
        try:
            device = torch.device("cuda")

            # 使用缓存机制加载模型
            model_a = Evaluator._load_model(model_a_path, device)
            model_b = Evaluator._load_model(model_b_path, device)

            current_model = np.random.choice([model_a, model_b])
            state = GameCore()

            while True:
                mcts = MCTS(current_model, cfg)
                pi = mcts.getActionProb(state)
                action_index = np.argmax(pi)
                next_state = GameCore(state)
                next_state.step(action_index)
                ended = next_state.is_terminal()

                if ended != 0:
                    winner = "黑胜" if (ended == 1 and next_state.current_player == 0) else "白胜"
                    a_win = (current_model == model_a and winner == "黑胜") or \
                            (current_model == model_b and winner == "白胜")
                    return 1 if a_win else 0

                state = next_state
                current_model = model_b if current_model == model_a else model_a
        finally:
            torch.cuda.empty_cache()

    @staticmethod
    def get_latest_checkpoint(checkpoint_dir):
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: datetime.strptime(x.split('_')[1] + '_' + x.split('_')[2], r"%Y%m%d_%H%M%S"), reverse=True)
        return os.path.join(checkpoint_dir, checkpoints[0])

    @staticmethod
    def _load_model(checkpoint_path, device):
        model = AlphaZeroNet().to(device)
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        return model.eval()
