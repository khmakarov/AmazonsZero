#pragma once
#include <array>
#include <cstdint>
#include <tuple>
#include <fstream>
#include <unordered_map>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
namespace py = pybind11;
using MoveAction = std::tuple<uint8_t, uint8_t, uint8_t>;
constexpr size_t TOTAL_ACTIONS = 33344, POSSIBLE_ACTIONS = 1500;
inline std::array<MoveAction, TOTAL_ACTIONS> action_list;
inline std::unordered_map<size_t, size_t> action_map;
void load_actions();
py::array_t<int> generate_mask_np(const std::pair<std::array<MoveAction, POSSIBLE_ACTIONS>, int> &legal_actions);