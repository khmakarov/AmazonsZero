import random
from unittest import result
import h5py
import multiprocessing as mp
import numpy as np
import orjson
from mpi4py import MPI
from pathlib import Path
from tqdm import tqdm
from core.cpp.build.Amazons import GameCore
from core.cpp.build.Amazons import Evaluate

PROCESS_DATA = Path("/home/khmakarov/AmazonsZero/data/pretrain")
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


class HDF5Saver:

    def __init__(self, path, chunk_size=8192):
        self.path = Path(path)
        self.chunk_size = chunk_size
        self.file = None
        self.states = None
        self.pis = None
        self.valids = None
        self.ended = None

        file_exists = self.path.exists()

        self.file = h5py.File(self.path, 'a' if file_exists else 'w', driver='mpio', comm=MPI.COMM_SELF, libver='latest')

        if not file_exists:
            self.states = self.file.create_dataset('states', shape=(0, 8, 8, 5), maxshape=(None, 8, 8, 5), dtype='int8', chunks=(chunk_size, 8, 8, 5), compression='gzip')
            self.pis = self.file.create_dataset('pis', shape=(0, 33344), maxshape=(None, 33344), dtype='float16', chunks=(chunk_size, 33344), compression='gzip')
            self.valids = self.file.create_dataset('valids', shape=(0, 1500), maxshape=(None, 1500), dtype='int32', chunks=(chunk_size, 1500), compression='gzip')
            self.ended = self.file.create_dataset('ended', shape=(0, ), maxshape=(None, ), dtype='int8', chunks=(chunk_size, ), compression='gzip')
        else:
            self.states = self.file['states']
            self.pis = self.file['pis']
            self.valids = self.file['valids']
            self.ended = self.file['ended']

        for dset in [self.states, self.pis, self.valids, self.ended]:
            dset.make_scale('batch_index')
            dset.dims[0].label = 'batch_index'

    def add_batch(self, states_batch, pis_batch, valids_batch, ended_batch):
        new_size = self.states.shape[0] + len(states_batch)
        self.states.resize((new_size, 8, 8, 5))
        self.pis.resize((new_size, 33344))
        self.valids.resize((new_size, 1500))
        self.ended.resize((new_size, ))

        self.states[-len(states_batch):] = states_batch
        self.pis[-len(pis_batch):] = pis_batch
        self.valids[-len(valids_batch):] = valids_batch
        self.ended[-len(ended_batch):] = ended_batch

    def close(self):
        if self.file:
            self.file.close()


def process_game(game_chunk):
    states = []
    pis = []
    valids_idxs = []
    values = []
    for game_data in game_chunk:
        state = GameCore()
        ended = 1.0 if game_data["scores"][0] else -1.0
        step = 0
        for entry in game_data.get("log", []):
            if "0" in entry or "1" in entry:
                player = "0" if "0" in entry else "1"
                action = entry[player]["response"]
                action_index = state.action2index(action["x0"], action["y0"], action["x1"], action["y1"], action["x2"], action["y2"])
                if player == "0" and ended == 1.0 or player == "1" and ended == -1.0:
                    if player == "0":
                        e = Evaluate(state.black, state.white, state.invert())
                    else:
                        e = Evaluate(state.white, state.black, state.invert())
                    pi = np.zeros(33344, dtype=np.float16)
                    pi[action_index] = 1.0
                    valids_idx = state.get_legal_actions_np()
                    if valids_idx[0] > 1:
                        states.append(state.get_state_np())
                        pis.append(pi)
                        valids_idxs.append(valids_idx)
                        values.append(e.eval())
                next_state = GameCore(state)
                next_state.step(action_index)
                step += 1
                state = next_state

    return states, pis, valids_idxs, values


def parallel_loader(current_folder):
    if rank == 0:
        all_files = list(current_folder.rglob("*.jsonl"))
        file_chunks = [all_files[i::size] for i in range(size)]
    else:
        file_chunks = None

    local_files = comm.scatter(file_chunks, root=0)
    local_data = []
    for file_path in local_files:
        with open(file_path, "rb") as f:
            local_data.extend(orjson.loads(line) for line in f)

    return local_data


def main():
    for match_num in range(1, 5):
        current_folder = PROCESS_DATA / f"Matches{match_num}"
        output_path = PROCESS_DATA / f"data{match_num}.h5"
        all_data = parallel_loader(current_folder)
        saver = HDF5Saver(output_path)
        total_subchunks = 10
        subchunk_size = len(all_data) // total_subchunks
        subchunks = [all_data[i * subchunk_size:(i + 1) * subchunk_size] for i in range(total_subchunks)]

        remainder = len(all_data) % total_subchunks
        if remainder:
            subchunks[-1].extend(all_data[-remainder:])

        with tqdm(subchunks, desc="总进度", unit="chunk", position=0) as main_pbar:
            for subchunk in main_pbar:
                sub_idx = main_pbar.n + 1

                num_processes = 8
                chunk_size = len(subchunk) // num_processes
                chunks = [subchunk[i * chunk_size:(i + 1) * chunk_size] for i in range(num_processes)]
                remainder_sub = len(subchunk) % num_processes
                if remainder_sub:
                    chunks[-1].extend(subchunk[-remainder_sub:])
                with mp.Pool(processes=num_processes) as pool:
                    results = pool.map(process_game, chunks)

                states_batch = []
                pis_batch = []
                valids_batch = []
                ended_batch = []
                for states, pis, valids, ended in results:
                    states_batch.extend(states)
                    pis_batch.extend(pis)
                    valids_batch.extend(valids)
                    ended_batch.extend(ended)
                saver.add_batch(np.stack(states_batch), np.stack(pis_batch), np.stack(valids_batch), np.stack(ended_batch))
                tqdm.write(f"子块 {sub_idx} 处理完成，已保存{saver.states.shape[0]}条数据")

        saver.close()


if __name__ == "__main__":
    main()
