#pragma once
#include "game_core.h"
#include <vector>
#include <memory>

class MCTSNode
{
public:
    MCTSNode(std::shared_ptr<GameCore> state, MCTSNode *parent = nullptr) : state(state), parent(parent), visit_count(0), value_sum(0), prior(0) {}

    MCTSNode *select(double c_puct);
    void expand(const std::vector<float> &action_probs);
    void update(float value);
    bool is_leaf() const;

    std::shared_ptr<GameCore> state;
    MCTSNode *parent;
    std::vector<std::pair<int, MCTSNode *>> children; // action_index -> node
    int visit_count;
    float value_sum;
    float prior;
};

class MCTSEngine
{
public:
    MCTSEngine(int n_simulations, double c_puct) : n_simulations(n_simulations), c_puct(c_puct) {}
    std::vector<float> run(std::shared_ptr<GameCore> root_state, py::function policy_value_fn);

private:
    void simulate(std::shared_ptr<GameCore> state);

    int n_simulations;
    double c_puct;
    std::vector<MCTSNode *> node_pool;
};
