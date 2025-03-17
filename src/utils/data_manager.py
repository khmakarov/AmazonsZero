#data_manager.py
import json
import random
import zlib
from src.utils.database import AmazonsDatabase
from core.cpp.build.Amazons import GameCore


class DataManager:

    def __init__(self):
        self.db = AmazonsDatabase()
        self.train_data = []

    def add_episode_data(self, episode_data, result):
        visual_data = [(DataManager._deserialize_state(s_bytes), a) for s_bytes, a, _ in episode_data]
        self.db.save_game(visual_data, result)

    def add_train_data(self, train_data):
        processed = [(DataManager._deserialize_state(s_bytes), pi, ended) for s_bytes, pi, ended in train_data]
        self.train_data.extend(processed)

    def clear_train_data(self):
        self.train_data.clear()

    def sample_batch(self, batch_size):
        return random.sample(self.train_data, batch_size)

    @staticmethod
    def _deserialize_state(data: bytes):
        """从字节流重建游戏状态"""
        state_data = json.loads(zlib.decompress(data).decode())

        # 重建GameCore实例
        state = GameCore()
        state.current_player = state_data["current_player"]
        state.black = state_data["black"]
        state.white = state_data["white"]
        state.blocks = state_data["blocks"]

        return state
