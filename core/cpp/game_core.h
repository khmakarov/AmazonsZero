#pragma once
#include "action_index.h"
#include <format>
#include <functional>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
class GameCore
{
public:
	GameCore();
	GameCore(const GameCore &) = default;

	std::string stringRepresentation() const;
	std::array<std::array<std::array<int, 5>, 8>, 8> get_state() const;						// 将对局状态输入python端
	std::pair<std::array<bool, TOTAL_ACTIONS>, std::array<int, 0x800>> get_legal_actions(); // 返回当前棋盘状态所有合法动作的掩码
	void step(int action_index);															// 应用行动
	int is_terminal() const;																// 0未结束,-1 current_player输,1 current_player赢
	std::tuple<uint8_t, uint8_t, uint8_t> index2action(int index);

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
	uint64_t black, white, blocks;
	uint64_t piece_from_backpack = 0, piece_to_backpack = 0;
	int current_player;

	uint64_t generate_moves(uint64_t from) const;
	static uint64_t ray_cast(uint64_t from, std::pair<uint64_t, int> dir_mask, uint64_t blanks);
	void apply_move(uint64_t from, uint64_t to);
	void place_block(uint64_t block);
	void restore_action();
	static std::tuple<uint8_t, uint8_t, uint8_t> pack_action(uint64_t from, uint64_t to, uint64_t block);
	static std::array<uint64_t, 3> unpack_action(const std::tuple<int, int, int> &action);
	static std::array<uint64_t, 4> unpack_pieces(uint64_t pieces);
	static uint64_t lowest1(uint64_t x);
	static int lowest1_bit(uint64_t x);
};