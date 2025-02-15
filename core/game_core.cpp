#include <chrono>
#include <cstdlib>
#include <intrin.h>
#include <iostream>
#include <tuple>
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
    /* 加载状态（Python接口）
    void load_state(py::array_t<uint8_t> state_np)
    {
        auto buf = state_np.request();
        // 将numpy数组转换为位棋盘表示
        // 实现细节需补充...
    }
    */
    std::vector<std::tuple<int, int, int>> get_legal_actions()
    {
        auto start_total = std::chrono::steady_clock::now();
        std::vector<std::tuple<int, int, int>> actions;
        actions.reserve(0x4D0);

        // Measure unpack_pieces time
        auto start_unpack = std::chrono::steady_clock::now();
        const std::array<uint64_t, 4> my_pieces = unpack_pieces(current_player ? white : black);
        auto end_unpack = std::chrono::steady_clock::now();
        profile_data.unpack_pieces_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_unpack - start_unpack);

        for (const auto &from : my_pieces)
        {
            // Measure generate_moves (piece) time
            auto start_gen_piece = std::chrono::steady_clock::now();
            uint64_t TO = generate_moves(from);
            auto end_gen_piece = std::chrono::steady_clock::now();
            profile_data.generate_moves_piece_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen_piece - start_gen_piece);

            for (uint64_t to; TO; TO ^= to)
            {
                auto start_lowest1_piece = std::chrono::steady_clock::now();
                to = lowest1(TO);
                auto end_lowest1_piece = std::chrono::steady_clock::now();
                profile_data.lowest1_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_lowest1_piece - start_lowest1_piece);
                // Measure apply_move time
                auto start_apply = std::chrono::steady_clock::now();
                apply_move(from, to);
                auto end_apply = std::chrono::steady_clock::now();
                profile_data.apply_move_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_apply - start_apply);

                // Measure generate_moves (block) time
                auto start_gen_block = std::chrono::steady_clock::now();
                uint64_t BLOCK = generate_moves(to);
                auto end_gen_block = std::chrono::steady_clock::now();
                profile_data.generate_moves_block_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_gen_block - start_gen_block);

                for (uint64_t block; BLOCK; BLOCK ^= block)
                {
                    auto start_lowest1_piece1 = std::chrono::steady_clock::now();
                    block = lowest1(BLOCK);
                    auto end_lowest1_piece1 = std::chrono::steady_clock::now();
                    profile_data.lowest1_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_lowest1_piece1 - start_lowest1_piece1);
                    auto start_push_back_action_piece = std::chrono::steady_clock::now();
                    actions.push_back(pack_action(from, to, block));
                    auto end_push_back_action_piece = std::chrono::steady_clock::now();
                    profile_data.push_back_action_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_push_back_action_piece - start_push_back_action_piece);
                }

                // Measure restore_action time
                auto start_restore = std::chrono::steady_clock::now();
                restore_action();
                auto end_restore = std::chrono::steady_clock::now();
                profile_data.restore_action_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_restore - start_restore);
            }
        }

        auto end_total = std::chrono::steady_clock::now();
        profile_data.get_legal_actions_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end_total - start_total);
        return actions;
    }

    void print_profile_data() const
    {
        std::cout << "=== Function Time Profile ===\n";
        std::cout << "get_legal_actions total: " << profile_data.get_legal_actions_total.count() / 1000 << " μs\n";
        std::cout << "unpack_pieces:           " << profile_data.unpack_pieces_time.count() / 1000 << " μs\n";
        std::cout << "generate_moves (piece):  " << profile_data.generate_moves_piece_time.count() / 1000 << " μs\n";
        std::cout << "generate_moves (block):  " << profile_data.generate_moves_block_time.count() / 1000 << " μs\n";
        std::cout << "ray_cast:                " << profile_data.ray_cast_time.count() / 1000 << " μs\n";
        std::cout << "apply_move:              " << profile_data.apply_move_time.count() / 1000 << " μs\n";
        std::cout << "restore_action:          " << profile_data.restore_action_time.count() / 1000 << " μs\n";
        std::cout << "push_back_action:        " << profile_data.push_back_action_time.count() / 1000 << " μs\n";
        std::cout << "lowest1:                 " << profile_data.lowest1_time.count() / 1000 << " μs\n";
        std::cout << "==============================\n";
    }
    void step(const std::tuple<int, int, int> &unpacked_action)
    {
        const std::array<uint64_t, 3> action = unpack_action(unpacked_action);
        apply_move(action[0], action[1]);
        place_block(action[2]);
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

    mutable struct ProfileData
    {
        std::chrono::nanoseconds get_legal_actions_total{0};
        std::chrono::nanoseconds unpack_pieces_time{0};
        std::chrono::nanoseconds generate_moves_piece_time{0};
        std::chrono::nanoseconds generate_moves_block_time{0};
        std::chrono::nanoseconds ray_cast_time{0};
        std::chrono::nanoseconds apply_move_time{0};
        std::chrono::nanoseconds restore_action_time{0};
        std::chrono::nanoseconds push_back_action_time{0};
        std::chrono::nanoseconds lowest1_time{0};
    } profile_data;

    uint64_t generate_moves(const uint64_t from) const
    {
        const uint64_t blanks = ~(black | white | blocks);
        uint64_t moves = from;
        for (const auto &dir : DIRECTION_MASKS)
        {
            auto start_ray = std::chrono::steady_clock::now();
            moves |= ray_cast(from, dir, blanks);
            auto end_ray = std::chrono::steady_clock::now();
            profile_data.ray_cast_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end_ray - start_ray);
        }
        return moves ^ from;
    }

    static uint64_t ray_cast(const uint64_t from, const std::pair<uint64_t, int> dir_mask, const uint64_t blanks)
    {
        uint64_t moves = from;
        for (int i = 0; i < 7; i++)
        {
            const uint64_t temp = moves;
            moves |= dir_mask.first & blanks & (dir_mask.second < 0 ? moves >> (-dir_mask.second) : moves << dir_mask.second);
            if (temp == moves)
                break;
        }
        return moves;
    }

    void apply_move(const uint64_t from, const uint64_t to)
    {
        auto start = std::chrono::steady_clock::now();
        current_player ? white = (white ^ from) | to : black = (black ^ from) | to;
        piece_from_backpack = from;
        piece_to_backpack = to;
        auto end = std::chrono::steady_clock::now();
        profile_data.apply_move_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
    }
    void place_block(const uint64_t block)
    {
        blocks |= block;
        blocks_backpack = block;
    }
    void restore_action()
    {
        auto start = std::chrono::steady_clock::now();
        current_player ? white = (white ^ piece_to_backpack) | piece_from_backpack : black = (black ^ piece_to_backpack) | piece_from_backpack;
        blocks ^= blocks_backpack;
        auto end = std::chrono::steady_clock::now();
        profile_data.restore_action_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
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
/*
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
*/
int main()
{
    GameCore game;
    game.get_legal_actions();
    game.print_profile_data();
    return 0;
}