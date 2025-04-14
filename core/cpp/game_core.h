#pragma once
#include <xxhash.h>
#include <functional>
#include "action_index.h"
class GameCore
{
public:
	GameCore();
	GameCore(const GameCore &) = default;

	int current_player;
	uint64_t black, white, blocks;

	size_t stringRepresentation() const;
	py::array_t<int> get_state_np() const;	 // 将对局状态输入python端
	py::array_t<int> get_legal_actions_np(); // 返回当前棋盘状态所有合法动作的掩码
	void step(int action_index);			 // 应用行动
	int is_terminal() const;				 // 0未结束,-1 current_player输,1 current_player赢
	MoveAction index2action(int index);

private:
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
	uint64_t piece_from_backpack = 0, piece_to_backpack = 0;

	uint64_t generate_moves(uint64_t from) const;
	static uint64_t ray_cast(uint64_t from, std::pair<uint64_t, int> dir_mask, uint64_t blanks);
	void apply_move(uint64_t from, uint64_t to);
	void restore_action();
	static MoveAction pack_action(uint64_t from, uint64_t to, uint64_t block);
	static std::array<uint64_t, 3> unpack_action(const std::tuple<int, int, int> &action);
	static std::array<uint64_t, 4> unpack_pieces(uint64_t pieces);
	static uint64_t lowest1(uint64_t x);
	static int lowest1_bit(uint64_t x);
};