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
