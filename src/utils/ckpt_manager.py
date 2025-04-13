#ckpt_manager.py
import os
import torch
from datetime import datetime


class CheckpointManager:

    def __init__(self):
        self.checkpoint_dir = "/home/khmakarov/AmazonsZero/model/checkpoint"
        self.best_model = None

    def save(self, model, optimizer, win_rate=float('nan')):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"checkpoint_{timestamp}_{win_rate:.2f}.pth"
        filepath = os.path.join(self.checkpoint_dir, filename)

        torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'win_rate': win_rate}, filepath)

        if not self.best_model or win_rate >= self.best_model[1]:
            self.best_model = (filepath, win_rate)

        return filepath

    def get_latest_checkpoint(self):
        checkpoints = [f for f in os.listdir(self.checkpoint_dir) if f.startswith("checkpoint_")]
        if not checkpoints:
            return None
        checkpoints.sort(key=lambda x: datetime.strptime(x.split('_')[1] + '_' + x.split('_')[2], "%Y%m%d_%H%M%S"), reverse=True)
        return os.path.join(self.checkpoint_dir, checkpoints[0])
