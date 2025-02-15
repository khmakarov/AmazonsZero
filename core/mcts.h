#pragma once
#include <pybind11/stl.h>
#include "game_core.h"

class MCTSNode
{
public:
    GameCore state;
    std::vector<std::tuple<int, int, int>> legal_actions;
    std::unordered_map<size_t, std::unique_ptr<MCTSNode>> children;
    float value_sum = 0, prior = 0;
    int visit_count = 0;

    MCTSNode(const GameCore &game);
    MCTSNode *select(double c_puct);
    void expand(const std::vector<float> &action_probs);
    void backup(float value);
};

class MCTSBridge
{
public:
    MCTSBridge(int n_simulations, double c_puct);
    std::vector<float> get_action_probs(py::object py_game, py::function policy_fn);
    void update_with_move(int last_move);

private:
    std::unique_ptr<MCTSNode> root;
    int n_simulations;
    double c_puct;
};
