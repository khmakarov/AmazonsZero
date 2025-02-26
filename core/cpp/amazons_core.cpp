#include "game_core.h"
PYBIND11_MODULE(amazons_core, m)
{
    load_actions();
    py::class_<GameCore>(m, "GameCore")
        .def(py::init<>())
        .def(py::init<const GameCore &>())
        .def_readwrite("current_player", &GameCore::current_player)
        .def("stringRepresentation", &GameCore::stringRepresentation)
        .def("get_legal_actions", &GameCore::get_legal_actions)
        .def("step", &GameCore::step)
        .def("is_terminal", &GameCore::is_terminal)
        .def("get_state", &GameCore::get_state);
}