#include "game_core.h"

GameCore::GameCore() : current_player(0),
					   black(0x0000000000810024),
					   white(0x2400810000000000),
					   blocks(0) {}

std::string GameCore::stringRepresentation() const
{
	size_t hash = 0;
	std::hash<uint64_t> hasher;
	hash ^= hasher(black) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
	hash ^= hasher(white) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
	hash ^= hasher(blocks) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
	hash ^= std::hash<int>()(current_player);
	std::string s = std::format("{}", hash);
	return s;
}

std::array<std::array<std::array<int, 5>, 8>, 8> GameCore::get_state() const
{
	std::array<std::array<std::array<int, 5>, 8>, 8> state{};
	const int player_layer = current_player == 0 ? 3 : 4;
	for (int y = 0, cnt = 0; y < 8; ++y)
	{
		for (int x = 0; x < 8; ++x)
		{
			uint64_t mask = 1ULL << (cnt++);
			auto &cell = state[y][x];
			if (blocks & mask)
				cell[2] = 1;
			else if (white & mask)
				cell[1] = 1;
			else if (black & mask)
				cell[0] = 1;
			cell[player_layer] = 1;
		}
	}
	return state;
}

std::pair<std::array<bool, TOTAL_ACTIONS>, std::array<int, POSSIBLE_ACTIONS>> GameCore::get_legal_actions()
{
	std::array<MoveAction, POSSIBLE_ACTIONS> actions;
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
	return generate_mask({actions, count});
}

void GameCore::step(const int action_index)
{
	const MoveAction &unpacked_action = action_list[action_index];
	const std::array<uint64_t, 3> action = unpack_action(unpacked_action);
	apply_move(action[0], action[1]);
	blocks |= action[2];
	current_player ^= 1;
}

int GameCore::is_terminal() const
{
	if (generate_moves(black) && generate_moves(white))
		return 0;
	if (generate_moves(current_player ? white : black))
		return 1;
	return -1;
}

MoveAction GameCore::index2action(const int index)
{
	const MoveAction &action = action_list[index];
	return {std::get<0>(action), std::get<1>(action), std::get<2>(action)};
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

void GameCore::restore_action()
{
	current_player ? white = (white ^ piece_to_backpack) | piece_from_backpack : black = (black ^ piece_to_backpack) | piece_from_backpack;
}

MoveAction GameCore::pack_action(const uint64_t from, const uint64_t to, const uint64_t block)
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

uint64_t GameCore::lowest1(const uint64_t x)
{
	return x & (-static_cast<int64_t>(x));
}

int GameCore::lowest1_bit(const uint64_t x)
{
	return static_cast<int>(__builtin_ctzll(x));
}