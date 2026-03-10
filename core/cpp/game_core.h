#pragma once
#include <array>
#include <iostream>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <immintrin.h>
#include <omp.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <tuple>
#include <unordered_map>
#include <xxhash.h>
namespace py = pybind11;
using MoveAction = std::tuple<uint8_t, uint8_t, uint8_t>;
using ChildState = std::tuple<py::array_t<size_t>, py::array_t<int8_t>, py::array_t<int>>;
constexpr size_t TOTAL_ACTIONS = 33344, POSSIBLE_ACTIONS = 1500;
inline std::array<MoveAction, TOTAL_ACTIONS> action_list;
inline std::unordered_map<size_t, size_t> action_map;
void load_actions();
class GameCore
{
public:
	GameCore();
	GameCore(const GameCore &) = default;

	int current_player;
	uint64_t black, white, blocks;
	uint64_t invert();
	size_t compute_state_hash() const;
	py::array_t<int8_t> get_state_np() const;					 // 将对局状态输入python端
	py::array_t<int> get_legal_actions_np();					 // 返回当前棋盘状态所有合法动作的掩码
	ChildState get_child_state_np(py::array_t<int> cut_indices); // 所有子状态的np数组和合法掩码
	void step(int action_index);								 // 应用行动
	int is_terminal();											 // 0未结束,1 黑方赢,-1 白方赢
	MoveAction index2action(int index);
	int action2index(int x0, int y0, int x1, int y1, int x2, int y2);

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
	alignas(32) static constexpr std::array<uint64_t, 64> MASK_TABLE = []()
	{
		std::array<uint64_t, 64> masks;
		for (int i = 0; i < 64; ++i)
			masks[i] = 1ULL << i;
		return masks;
	}();

	uint64_t piece_from_backpack = 0, piece_to_backpack = 0;

	uint64_t generate_moves(uint64_t from) const;
	static uint64_t ray_cast(uint64_t from, std::pair<uint64_t, int> dir_mask, uint64_t blanks);
	void fill_legal_actions(int *target, int &count);
	void apply_move(uint64_t from, uint64_t to);
	void restore_action();
	static std::array<uint64_t, 4> unpack_pieces(uint64_t pieces);
	static uint64_t lowest1(uint64_t x);
	static int lowest1_bit(uint64_t x);
};