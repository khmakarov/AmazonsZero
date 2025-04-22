#include "game_core.h"

void load_actions()
{
	int count = 0;
	std::ifstream f("/home/khmakarov/AmazonsZero/data/action_space/amazons_actions.bin", std::ios::binary);
	while (true)
	{
		MoveAction action;
		f.read(reinterpret_cast<char *>(&action), sizeof(MoveAction));
		if (f.eof())
			break;
		action_list[count] = action;
		action_map[static_cast<size_t>(std::get<0>(action)) << 16 | (static_cast<size_t>(std::get<1>(action)) << 8) | static_cast<size_t>(std::get<2>(action))] = count++;
	}
}

GameCore::GameCore() : current_player(0),
					   black(0x0000000000810024),
					   white(0x2400810000000000),
					   blocks(0) {}

size_t GameCore::compute_state_hash() const
{
	XXH64_state_t *state = XXH64_createState();
	XXH64_reset(state, 0);
	XXH64_update(state, &black, sizeof(black));
	XXH64_update(state, &white, sizeof(white));
	XXH64_update(state, &blocks, sizeof(blocks));
	XXH64_update(state, &current_player, sizeof(current_player));
	XXH64_hash_t hash = XXH64_digest(state);
	XXH64_freeState(state);
	return hash;
}

py::array_t<int8_t> GameCore::get_state_np() const
{
	const int player_layer = (current_player == 0) ? 3 : 4;
	constexpr size_t alignment = 32;
	const size_t num_elements = 320;
	void *raw_mem = aligned_alloc(alignment, num_elements * sizeof(int8_t));
	auto deleter = [](void *p)
	{ free(p); };
	std::unique_ptr<int8_t[], decltype(deleter)> data(static_cast<int8_t *>(raw_mem), deleter);
	memset(data.get(), 0, num_elements);
	int8_t *ptr = data.get();
	py::capsule capsule(ptr, deleter);
	data.release();
	py::array_t<int8_t> state({8, 8, 5}, ptr, capsule);
	auto buf_state = state.mutable_unchecked<3>();
	for (int y = 0, cnt = 0; y < 8; ++y)
	{
		for (int x = 0; x < 8; ++x, ++cnt)
		{
			const uint64_t mask = MASK_TABLE[cnt];
			buf_state(y, x, 0) = (black & mask) ? 1 : 0;
			buf_state(y, x, 1) = (white & mask) ? 1 : 0;
			buf_state(y, x, 2) = (blocks & mask) ? 1 : 0;
			buf_state(y, x, player_layer) = 1;
		}
	}
	return state;
}

py::array_t<int> GameCore::get_legal_actions_np()
{
	int count = 0;
	std::vector<int> data(POSSIBLE_ACTIONS, -1);
	py::array_t<int> mask_index(POSSIBLE_ACTIONS, data.data());
	auto buf = mask_index.mutable_unchecked<1>();
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
				auto action_sum = static_cast<size_t>(lowest1_bit(from) << 16 | (lowest1_bit(to) << 8) | lowest1_bit(block));
				buf[++count] = action_map.find(action_sum)->second;
			}
			restore_action();
		}
	}
	buf[0] = count;
	return mask_index;
}

ChildState GameCore::get_child_state_np(py::array_t<int> valids_idx)
{
	auto buf = valids_idx.unchecked<1>();
	const int num_child = buf(0);
	std::vector<size_t> child_hash(num_child, 0);
	std::vector<int> child_valids_idx(num_child * POSSIBLE_ACTIONS, -1);

	auto child_hash_np = py::array_t<size_t>(num_child, child_hash.data());
	constexpr size_t alignment = 32;
	const size_t num_elements = num_child * 320;
	void *raw_mem = aligned_alloc(alignment, num_elements * sizeof(int8_t));
	auto deleter = [](void *p)
	{ free(p); };
	std::unique_ptr<int8_t[], decltype(deleter)> data(static_cast<int8_t *>(raw_mem), deleter);
	memset(data.get(), 0, num_elements);
	int8_t *ptr = data.get();
	py::capsule capsule(ptr, deleter);
	data.release();
	auto child_states_np = py::array_t<int8_t>({num_child, 8, 8, 5}, ptr, capsule);
	auto child_valids_idx_np = py::array_t<int>({num_child, 1500}, child_valids_idx.data());

	auto buf_hash = child_hash_np.mutable_unchecked<1>();
	auto buf_states = child_states_np.mutable_unchecked<4>();
	auto buf_valids = child_valids_idx_np.mutable_unchecked<2>();
#pragma omp parallel for schedule(dynamic) if (num_child > 100)
	for (int i = 0; i < num_child; ++i)
	{
		GameCore child(*this);
		child.step(buf(i + 1));
		buf_hash[i] = child.compute_state_hash();
		const int player_layer = (child.current_player == 0) ? 3 : 4;
		for (int y = 0, cnt = 0; y < 8; ++y)
		{
			for (int x = 0; x < 8; ++x, ++cnt)
			{
				const uint64_t mask = MASK_TABLE[cnt];
				buf_states(i, y, x, 0) = (child.black & mask) ? 1 : 0;
				buf_states(i, y, x, 1) = (child.white & mask) ? 1 : 0;
				buf_states(i, y, x, 2) = (child.blocks & mask) ? 1 : 0;
				buf_states(i, y, x, player_layer) = 1;
			}
		}
		int num_legal = 0;
		child.fill_legal_actions(&buf_valids(i, 1), num_legal);
		buf_valids(i, 0) = num_legal;
	}
	return std::make_tuple(child_hash_np, child_states_np, child_valids_idx_np);
}

void GameCore::step(const int action_index)
{
	const MoveAction &unpacked_action = action_list[action_index];
	const std::array<uint64_t, 3> action = unpack_action(unpacked_action);
	apply_move(action[0], action[1]);
	blocks |= action[2];
	current_player ^= 1;
}

int GameCore::is_terminal()
{
	if (generate_moves(current_player ? white : black))
		return 0;
	else
		return current_player ? 1 : -1;
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

void GameCore::fill_legal_actions(int *target, int &count)
{
	count = 0;
	const std::array<uint64_t, 4> my_pieces = unpack_pieces(current_player ? white : black);
	for (const auto &from : my_pieces)
	{
		for (uint64_t TO = generate_moves(from), to; TO; TO ^= to)
		{
			to = lowest1(TO);
			apply_move(from, to);
			for (uint64_t BLOCK = generate_moves(to), block; BLOCK; BLOCK ^= block, count++)
			{
				block = lowest1(BLOCK);
				auto action_sum = static_cast<size_t>(lowest1_bit(from) << 16 | (lowest1_bit(to) << 8) | lowest1_bit(block));
				if (auto it = action_map.find(action_sum); it != action_map.end())
					target[count] = static_cast<int>(it->second);
			}
			restore_action();
		}
	}
}

void GameCore::apply_move(const uint64_t from, const uint64_t to)
{
	current_player ? white = (white ^ from) | to : black = (black ^ from) | to;
	piece_from_backpack = from, piece_to_backpack = to;
}

void GameCore::restore_action()
{
	current_player ? white = (white ^ piece_to_backpack) | piece_from_backpack : black = (black ^ piece_to_backpack) | piece_from_backpack;
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
	return __builtin_ctzll(x);
}