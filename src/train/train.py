# train.py
import os
import pstats
import hydra
import cProfile
import numpy as np
import torch
import torch.multiprocessing as mp
from torch.optim import Adam
from core.cpp.build.Amazons import GameCore
from core.python.mcts import MCTS
from core.python.net import AlphaZeroNet
from src.utils.ckpt_manager import CheckpointManager
from src.utils.data_manager import DataManager
from evaluate import Evaluator
from torch.utils.tensorboard.writer import SummaryWriter


class Trainer:

    def __init__(self, cfg):
        self.device = torch.device("cuda")
        self.load_model = cfg.load_model
        self.mcts = cfg.mcts
        self.nnet = self._init_net()
        self.training = cfg.training
        self.win_rate = cfg.evaluator.win_rate
        self.optimizer = Adam(self.nnet.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay)
        if self.load_model:
            checkpoint = torch.load(self.load_model)
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ckpt_mgr = CheckpointManager()
        self.data_mgr = DataManager()
        self.evaluator = Evaluator(cfg)
        self.writer = SummaryWriter(cfg.log_dir)
        self.iteration = 0
        self.writer.add_graph(self.nnet, torch.randn(1, 5, 8, 8).to(self.device))

    def _init_net(self):
        nnet = AlphaZeroNet().to(self.device)
        if self.load_model:
            checkpoint = torch.load(self.load_model)
            nnet.load_state_dict(checkpoint['model'])
        return nnet

    def learn(self):
        for i in range(1, self.training.num_iters + 1):
            print(f"Iteration {i}")
            self.iteration = i
            self._process_episodes(self._generate_episodes())
            self.data_mgr.flush_visual_data()
            if len(self.data_mgr.train_data) >= self.training.batch_size:
                self._train_step()
            if i % self.training.eval_freq == 0:
                self._evaluation_step()

    def _generate_episodes(self):
        nnet_state_dict = self.nnet.to("cpu").state_dict()
        self.nnet.to(self.device)
        with mp.Pool(processes=8) as pool:
            tasks = [(nnet_state_dict, self.mcts)] * self.training.num_eps
            results = pool.starmap(self.execute_episode_worker, tasks)
            pool.close()
            pool.join()
        return results

    def _process_episodes(self, results):
        for episode_data, ended in results:
            train_data, visual_data = zip(*[((x[0], x[1], x[2], ended), (x[0], x[3])) for x in episode_data])
            self.data_mgr.add_train_data(train_data)
            self.data_mgr.add_visual_data(visual_data, ended, self.iteration, 0)
        print("该迭代数据保存完毕")

    def _train_step(self):
        batches = self.data_mgr.sample_batch(self.training.batch_size)
        for batch in batches:
            total_loss, loss_pi, loss_v = self.train(batch)
            print(f"Loss/total:{total_loss}, Loss/policy:{loss_pi}, Loss/value:{loss_v}")
            self.writer.add_scalar('Loss/total', total_loss, self.iteration)
            self.writer.add_scalar('Loss/policy', loss_pi, self.iteration)
            self.writer.add_scalar('Loss/value', loss_v, self.iteration)
        print("该迭代数据训练完毕")

    def _evaluation_step(self):
        latest_path = self.ckpt_mgr.get_latest_checkpoint()
        current_path = self.ckpt_mgr.save(self.nnet, self.optimizer)

        if not latest_path:
            print(f"Initial model saved: {current_path}")
            return
        if hasattr(self, 'nnet'):
            del self.nnet
            torch.cuda.empty_cache()

        win_rate = self.evaluator.evaluate(self.iteration, current_path, latest_path)
        self.nnet = self._init_net()

        if win_rate >= self.win_rate:
            checkpoint = torch.load(current_path)
            self.nnet.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            best_model_path = self.ckpt_mgr.save(self.nnet, self.optimizer, win_rate)
            self.data_mgr.clear_train_data()
            print(f"New best model: {best_model_path} (Win rate: {win_rate:.2f})")
        else:
            best_model_path = latest_path
            checkpoint = torch.load(best_model_path)
            self.nnet.load_state_dict(checkpoint['model'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"Current model rejected. Win rate: {win_rate:.2f}")

        os.remove(current_path)
        self.writer.add_scalar('Evaluation/win_rate', win_rate, self.iteration)

    def train(self, batch):
        self.nnet.train()

        states, pis, valids_idx, vs = list(zip(*batch))
        states = torch.as_tensor(np.stack(states), dtype=torch.float32)
        states = states.permute(0, 3, 1, 2).to(self.device)
        target_pis = torch.tensor(np.array(pis), dtype=torch.float32).to(self.device)
        target_vs = torch.tensor(np.array(vs), dtype=torch.float32).unsqueeze(1).to(self.device)
        valids_idx = torch.as_tensor(np.stack(valids_idx), dtype=torch.int, device=self.device)

        out_pi, out_v = self.nnet(states, valids_idx)
        loss_pi = -torch.sum(target_pis * out_pi) / target_pis.size()[0]
        loss_v = torch.mean((target_vs - out_v)**2)
        total_loss = loss_pi + loss_v

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        torch.cuda.empty_cache()
        for name, param in self.nnet.named_parameters():
            self.writer.add_histogram(f'Parameters/{name}', param, self.iteration)
            if param.grad is not None:
                self.writer.add_histogram(f'Gradients/{name}', param.grad, self.iteration)

        return total_loss.item(), loss_pi.item(), loss_v.item()

    @staticmethod
    def execute_episode_worker(nnet_state_dict, cfg):
        try:
            episode_data = []
            nnet = AlphaZeroNet().to("cuda")
            nnet.load_state_dict(nnet_state_dict)
            mcts = MCTS(nnet, cfg)
            state = GameCore()
            pr = cProfile.Profile()
            pr.enable()
            while True:
                ended = state.is_terminal()
                if ended == 0:
                    pi, valids_idx = mcts.getActionProb(state)
                    action_index = np.random.choice(len(pi), p=pi)
                    next_state = GameCore(state)
                    next_state.step(action_index)
                    episode_data.append([state.get_state_np(), pi, valids_idx, state.index2action(action_index)])
                    state = next_state
                else:
                    episode_data.append(
                        [
                            state.get_state_np(),
                            np.zeros(cfg.TOTAL_ACTIONS, dtype=np.float64),
                            np.zeros(cfg.POSSIBLE_ACTIONS, dtype=np.int32),
                            state.index2action(0)
                        ]
                    )
                    break
            return (episode_data, ended)
        finally:
            pr.disable()
            stats = pstats.Stats(pr)
            stats.strip_dirs()  # 移除文件路径前缀
            stats.sort_stats('cumtime')  # 按累计时间排序
            stats.print_stats(r'mcts\.py|net\.py|train\.py')  # 仅显示指定文件
            del mcts, nnet
            torch.cuda.empty_cache()

    def __del__(self):
        self.writer.close()
        torch.cuda.empty_cache()


@hydra.main(version_base=None, config_path="../../", config_name="config")
def main(cfg) -> None:
    trainer = Trainer(cfg)
    trainer.learn()


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    mp.set_start_method('spawn', force=True)
    main()
