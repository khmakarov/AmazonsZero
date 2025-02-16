#include "game_core.h"
#include "mcts.h"

GameCore::GameCore() : black(0x0000000000810024),
                       white(0x2400810000000000),
                       blocks(0),
                       current_player(0)
{
}

GameCore GameCore::clone() const
{
    GameCore new_game;
    new_game.black = black;
    new_game.white = white;
    new_game.blocks = blocks;
    new_game.current_player = current_player;
    return new_game;
}

void GameCore::load_state(py::array_t<uint8_t> state_np)
{
    auto buf = state_np.request();
}

int GameCore::get_current_player() const
{
    return current_player;
}

std::pair<std::array<std::tuple<int, int, int>, 0x4D0>, int> GameCore::get_legal_actions()
{
    std::array<std::tuple<int, int, int>, 0x4D0> actions;
    const std::array<uint64_t, 4> my_pieces = unpack_pieces(current_player ? white : black);
    int count = 0;
    for (const auto &from : my_pieces)
    {
        for (uint64_t TO = generate_moves(from), to; TO; TO ^= to)
        {
            to = lowest1(TO);
            apply_move(from, to);
            for (uint64_t BLOCK = generate_moves(to), block; BLOCK; BLOCK ^= block, count++)
            {
                block = lowest1(BLOCK);
                actions[count] = pack_action(from, to, block);
            }
            restore_action();
        }
    }
    return {actions, count};
}

void GameCore::step(const std::tuple<int, int, int> &unpacked_action)
{
    const std::array<uint64_t, 3> action = unpack_action(unpacked_action);
    apply_move(action[0], action[1]);
    place_block(action[2]);
    current_player ^= 1;
}

bool GameCore::is_terminal() const
{
    return generate_moves(current_player ? white : black) == 0ull;
}

int GameCore::get_result() const
{
    return current_player ^ 1;
}

uint64_t GameCore::generate_moves(const uint64_t from) const
{
    const uint64_t blanks = ~(black | white | blocks);
    uint64_t moves = from;
    for (const auto &dir : DIRECTION_MASKS)
        moves |= ray_cast(from, dir, blanks);
    return moves ^ from;
}

uint64_t GameCore::ray_cast(const uint64_t from, const std::pair<uint64_t, int> dir_mask, const uint64_t blanks)
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

void GameCore::apply_move(const uint64_t from, const uint64_t to)
{
    current_player ? white = (white ^ from) | to : black = (black ^ from) | to;
    piece_from_backpack = from;
    piece_to_backpack = to;
}

void GameCore::place_block(const uint64_t block)
{
    blocks |= block;
    blocks_backpack = block;
}

void GameCore::restore_action()
{
    current_player ? white = (white ^ piece_to_backpack) | piece_from_backpack : black = (black ^ piece_to_backpack) | piece_from_backpack;
    blocks ^= blocks_backpack;
}

std::tuple<int, int, int> GameCore::pack_action(const uint64_t from, const uint64_t to, const uint64_t block)
{
    return {lowest1_bit(from), lowest1_bit(to), lowest1_bit(block)};
}

std::array<uint64_t, 3> GameCore::unpack_action(const std::tuple<int, int, int> &action)
{
    return {1ull << std::get<0>(action), 1ull << std::get<1>(action), 1ull << std::get<2>(action)};
}

std::array<uint64_t, 4> GameCore::unpack_pieces(const uint64_t pieces)
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

uint64_t GameCore::lowest1(uint64_t x)
{
    return x & (-static_cast<int64_t>(x));
}

int GameCore::lowest1_bit(uint64_t x)
{
    unsigned long index;
    _BitScanForward64(&index, x);
    return static_cast<int>(index);
}

PYBIND11_MODULE(libamazons, m)
{
    py::class_<GameCore>(m, "GameCore")
        .def(py::init<>())
        .def("load_state", &GameCore::load_state)
        .def("get_legal_actions", &GameCore::get_legal_actions)
        .def("step", &GameCore::step)
        .def("is_terminal", &GameCore::is_terminal)
        .def("get_result", &GameCore::get_result);

    py::class_<MCTSEngine>(m, "MCTSEngine")
        .def(py::init<int, double>())
        .def("run", &MCTSEngine::run);

    py::class_<MCTSNode>(m, "MCTSNode")
        .def("select", &MCTSNode::select)
        .def("expand", &MCTSNode::expand)
        .def("update", &MCTSNode::update);
}