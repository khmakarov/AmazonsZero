#pragma once
#include <array>
#include <tuple>
#include <fstream>
#include <unordered_map>
constexpr size_t TOTAL_ACTIONS = 33344;
inline std::array<std::tuple<uint8_t, uint8_t, uint8_t>, TOTAL_ACTIONS> action_list;
inline std::unordered_map<size_t, size_t> action_map;
void load_actions();
std::pair<std::array<bool, TOTAL_ACTIONS>, std::array<int, 0x800>> generate_mask(const std::pair<std::array<std::tuple<uint8_t, uint8_t, uint8_t>, 0x800>, int> &legal_actions);