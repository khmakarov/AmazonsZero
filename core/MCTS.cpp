// mcts.h
#pragma once
#include "game_core.h"

class MCTSNode
{
public:
    MCTSNode(GameCore state, MCTSNode *parent = nullptr);

    // UCT选择最佳子节点
    MCTSNode *select(double c_puct);

    // 扩展新节点
    void expand(const std::vector<float> &policy);

    // 回溯更新
    void backup(float value);

    // 获取访问次数分布
    std::vector<float> get_action_probs(float temperature);

    GameCore state;
    std::vector<std::pair<Action, MCTSNode *>> children;
    float value_sum = 0;
    int visit_count = 0;
    float prior = 0;
};

class MCTSSearcher
{
public:
    MCTSSearcher(int n_threads);

    // 并行搜索入口
    std::vector<float> run(const GameCore &root_state,
                           int simulations,
                           py::function policy_fn);

private:
    void search_thread(int thread_id);

    std::vector<MCTSNode *> roots;
    std::vector<std::thread> workers;
    std::mutex mutex;
};