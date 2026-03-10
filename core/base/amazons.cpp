#include "game.h"
PYBIND11_MODULE(Amazons, m)
{
    load_actions();
    py::class_<Game>(m, "Game")
        .def(py::init<>())
        .def("step", &Game::step);
}