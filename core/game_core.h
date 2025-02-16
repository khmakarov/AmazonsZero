#pragma once
#include <cstdlib>
#include <intrin.h>
#include <iostream>
#include <tuple>
#include <array>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "mcts.h"

namespace py = pybind11;

class GameCore
{
public:
    static constexpr std::array<std::pair<uint64_t, int>, 8> DIRECTION_MASKS = {{
        {0x7F7F7F7F7F7F7F7F, -1}, // West
        {0xFEFEFEFEFEFEFEFE, 1},  // East
        {0x00FFFFFFFFFFFFFF, -8}, // North
        {0xFFFFFFFFFFFFFF00, 8},  // South
        {0x007F7F7F7F7F7F7F, -9}, // NorthWest
        {0xFEFEFEFEFEFEFE00, 9},  // SouthEast
        {0x00FEFEFEFEFEFEFE, -7}, // NorthEast
        {0x7F7F7F7F7F7F7F00, 7},  // SouthWest
    }};

    GameCore();
    GameCore clone() const;

    void load_state(py::array_t<uint8_t> state_np);
    int get_current_player() const;
    std::pair<std::array<std::tuple<int, int, int>, 0x4D0>, int> get_legal_actions();
    void step(const std::tuple<int, int, int> &unpacked_action);
    bool is_terminal() const;
    int get_result() const;

private:
    uint64_t black, white, blocks;
    uint64_t piece_from_backpack = 0, piece_to_backpack = 0, blocks_backpack = 0;
    int current_player;

    uint64_t generate_moves(uint64_t from) const;
    static uint64_t ray_cast(uint64_t from, std::pair<uint64_t, int> dir_mask, uint64_t blanks);
    void apply_move(uint64_t from, uint64_t to);
    void place_block(uint64_t block);
    void restore_action();
    static std::tuple<int, int, int> pack_action(uint64_t from, uint64_t to, uint64_t block);
    static std::array<uint64_t, 3> unpack_action(const std::tuple<int, int, int> &action);
    static std::array<uint64_t, 4> unpack_pieces(uint64_t pieces);
    static uint64_t lowest1(uint64_t x);
    static int lowest1_bit(uint64_t x);
};