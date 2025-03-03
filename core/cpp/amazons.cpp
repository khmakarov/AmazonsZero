#include "game_core.h"
PYBIND11_MODULE(Amazons, m)
{
    load_actions();
    py::class_<GameCore>(m, "GameCore")
        .def(py::init<>())
        .def(py::init<const GameCore &>())
        .def("stringRepresentation", &GameCore::stringRepresentation)
        .def("get_state", &GameCore::get_state)
        .def("get_legal_actions", &GameCore::get_legal_actions)
        .def("step", &GameCore::step)
        .def("is_terminal", &GameCore::is_terminal)
        .def("index2action", &GameCore::index2action);
}