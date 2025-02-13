#include <iostream>
#include <cstdlib>
#include <tuple>
#include <intrin.h>
#include <chrono>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
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
    GameCore() : black(0x0000000000810024),
                 white(0x2400810000000000),
                 blocks(0),
                 current_player(0) {}
    GameCore clone() const
    {
        GameCore new_game;
        new_game.black = black;
        new_game.white = white;
        new_game.blocks = blocks;
        new_game.current_player = current_player;
        return new_game;
    }
    // 加载状态（Python接口）
    void load_state(py::array_t<uint8_t> state_np)
    {
        auto buf = state_np.request();
        // 将numpy数组转换为位棋盘表示
        // 实现细节需补充...
    }

    std::vector<std::tuple<int, int, int>> get_legal_actions()
    {
        std::vector<std::tuple<int, int, int>> actions;
        actions.reserve(0x4D0);
        const std::array<uint64_t, 4> my_pieces = unpack_pieces(current_player ? white : black);
        for (const auto &from : my_pieces)
        {
            for (uint64_t TO = generate_moves(from), to; TO; TO ^= to)
            {
                to = lowest1(TO);
                apply_move(from, to);
                for (uint64_t BLOCK = generate_moves(to), block; BLOCK; BLOCK ^= block)
                {
                    block = lowest1(BLOCK);
                    actions.push_back(pack_action(from, to, block));
                }
                restore_action();
            }
        }
        return actions;
    }
    void step(const std::tuple<int, int, int> &unpacked_action)
    {
        const std::array<uint64_t, 3> action = unpack_action(unpacked_action);
        const uint64_t from = action[0], to = action[1], block = action[2];
        apply_move(from, to);
        place_block(block);
        current_player ^= 1;
    }
    bool is_terminal() const
    {
        return generate_moves(current_player ? white : black) == 0ull;
    }
    int get_result() const
    {
        return current_player ^ 1;
    }

private:
    uint64_t black, white, blocks;
    uint64_t piece_from_backpack = 0, piece_to_backpack = 0, blocks_backpack = 0;
    int current_player;

    uint64_t generate_moves(const uint64_t from) const
    {
        const uint64_t blanks = ~(black | white | blocks);
        uint64_t moves = from;
        for (const auto &dir : DIRECTION_MASKS)
            moves |= ray_cast(from, dir, blanks);
        return moves ^ from;
    }
    static uint64_t ray_cast(const uint64_t from, const std::pair<uint64_t, int> dir_mask, const uint64_t blanks)
    {
        uint64_t moves = from;
        for (int i = 0; i < 7; i++)
        {
            const uint64_t temp = moves;
            moves |= dir_mask.first & blanks & (dir_mask.second < 0 ? moves >> (-dir_mask.second) : moves << dir_mask.second);
            // moves |= dir_mask.first & blanks & (dir_mask.second < 0 ? __ull_rshift(moves, -dir_mask.second) : __ll_lshift(moves, dir_mask.second));
            if (temp == moves)
                break;
        }
        return moves;
    }
    void apply_move(const uint64_t from, const uint64_t to)
    {
        current_player ? white = (white ^ from) | to : black = (black ^ from) | to;
        piece_from_backpack = from;
        piece_to_backpack = to;
    }
    void place_block(const uint64_t block)
    {
        blocks |= block;
        blocks_backpack = block;
    }
    void restore_action()
    {
        current_player ? white = (white ^ piece_to_backpack) | piece_to_backpack : black = (black ^ piece_to_backpack) | piece_from_backpack;
        blocks ^= blocks_backpack;
    }
    static std::tuple<int, int, int> pack_action(const uint64_t from, const uint64_t to, const uint64_t block)
    {
        return {lowest1_bit(from), lowest1_bit(to), lowest1_bit(block)};
    }
    static std::array<uint64_t, 3> unpack_action(const std::tuple<int, int, int> &action)
    {
        return {
            1ull << std::get<0>(action), 1ull << std::get<1>(action), 1ull << std::get<2>(action)};
    }
    static std::array<uint64_t, 4> unpack_pieces(const uint64_t pieces)
    {
        uint64_t temp = pieces;
        const uint64_t p1 = lowest1(temp);
        temp ^= p1;
        const uint64_t p2 = lowest1(temp);
        temp ^= p2;
        const uint64_t p3 = lowest1(temp);
        temp ^= p3;
        const uint64_t p4 = lowest1(temp);
        return {p1, p2, p3, p4};
    }
    static uint64_t lowest1(uint64_t x)
    {
        return x & (-static_cast<int64_t>(x));
    }
    static int lowest1_bit(uint64_t x)
    {
        unsigned long index;
        _BitScanForward64(&index, x);
        return static_cast<int>(index);
    }
};

PYBIND11_MODULE(libamazons, m)
{
    py::class_<GameCore>(m, "GameCore")
        .def(py::init<>())
        .def("load_state", &GameCore::load_state)
        .def("get_legal_actions", &GameCore::get_legal_actions)
        .def("step", &GameCore::step)
        .def("is_terminal", &GameCore::is_terminal)
        .def("get_result", &GameCore::get_result);
}

int main()
{
    GameCore game;
    auto start = std::chrono::steady_clock::now();
    game.get_legal_actions();
    auto end = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Time elapsed: " << duration << " microseconds" << std::endl;
    return 0;
}