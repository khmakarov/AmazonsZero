import numpy as np
from Amazons import GameCore


def main():
    print("a")
    game = GameCore()
    print(isinstance(np.array(game.get_state()), np.ndarray))
    valids, valids_idx = game.get_legal_actions()
    print(isinstance(valids_idx, np.ndarray))
    print(isinstance(valids, np.ndarray))


if __name__ == "__main__":
    main()
