import numpy as np
import amazons_core


def main():
    game = amazons_core.GameCore()
    print(game.stringRepresentation())
    game1 = amazons_core.GameCore(game)
    game1.step(999)
    print(game1.stringRepresentation())


if __name__ == "__main__":
    main()
