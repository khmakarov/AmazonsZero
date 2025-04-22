#include "game_core.h"
PYBIND11_MODULE(Amazons, m)
{
    load_actions();
    py::class_<GameCore>(m, "GameCore")
        .def(py::init<>())
        .def(py::init<const GameCore &>())
        .def_readwrite("current_player", &GameCore::current_player)
        .def("compute_state_hash", &GameCore::compute_state_hash)
        .def("get_state_np", &GameCore::get_state_np)
        .def("get_legal_actions_np", &GameCore::get_legal_actions_np)
        .def("get_child_state_np", &GameCore::get_child_state_np)
        .def("step", &GameCore::step)
        .def("is_terminal", &GameCore::is_terminal)
        .def("index2action", &GameCore::index2action);
}