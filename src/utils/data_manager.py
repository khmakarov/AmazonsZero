#data_manager.py
import random
from src.utils.database import AmazonsDatabase


class DataManager:

    def __init__(self):
        self.db = AmazonsDatabase()
        self.visual_buffer = []
        self.train_data = []

    def add_visual_data(self, episode_data, result):
        visual_data = [(s_bytes, a) for s_bytes, a, _, _ in episode_data]
        self.visual_buffer.append((visual_data, result))

    def add_train_data(self, train_data):
        processed = [(s_bytes, pi, valids_idx, ended) for s_bytes, pi, valids_idx, ended in train_data]
        self.train_data.extend(processed)

    def flush_visual_data(self):
        print("自对弈数据处理完毕")
        if not self.visual_buffer:
            return
        self.db.save_games(self.visual_buffer)
        self.visual_buffer.clear()

    def clear_train_data(self):
        self.train_data.clear()

    def sample_batch(self, batch_size):
        return random.sample(self.train_data, batch_size)
