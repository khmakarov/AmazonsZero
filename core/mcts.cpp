// mcts.cpp
#include "mcts.h"

MCTSNode *MCTSNode::select(double c_puct)
{
    MCTSNode *best_node = nullptr;
    double best_score = -INFINITY;

    for (auto &[action, child] : children)
    {
        double score = child->value_sum / child->visit_count + c_puct * child->prior * sqrt(visit_count) / (1 + child->visit_count);
        if (score > best_score)
        {
            best_score = score;
            best_node = child;
        }
    }
    return best_node;
}

void MCTSNode::expand(const std::vector<float> &action_probs)
{
    auto [legal_actions, count] = state->get_legal_actions();
    for (int i = 0; i < count; ++i)
    {
        auto new_state = std::make_shared<GameCore>(*state);
        new_state->step(legal_actions[i]);
        children.emplace_back(i, new MCTSNode(new_state, this));
        children.back().second->prior = action_probs[i];
    }
}

void MCTSNode::update(const float value)
{
    visit_count++;
    value_sum += value;
    if (parent)
        parent->update(value);
}
bool MCTSNode::is_leaf() const
{
    return children.empty();
}
/*run为整个mcts的总控调度循环,进行调用simulate并行化搜索,
 *simulate流程为:
 *1.selct:解析自己所有的子节点,通过uct算法选择出一个最优子节点
 *2.若该子节点不是终局,则调用python为该节点的所有子节点分配先验概率,然后为该节点更新value
 *3.若为终局则根据输赢更新value
 *最后run函数返回一个基于访问次数的策略向量
 */
std::vector<float> MCTSEngine::run(std::shared_ptr<GameCore> root_state, py::function policy_value_fn)
{
    MCTSNode root(root_state);

    for (int i = 0; i < n_simulations; ++i) //暂时使用单线程
        simulate(root_state, policy_value_fn);

    // 提取策略
    std::vector<float> counts(root.children.size(), 0);
    for (auto &[action, child] : root.children)
        counts[action] = child->visit_count;
    return counts;
}

void MCTSEngine::simulate(std::shared_ptr<GameCore> state, py::function policy_value_fn)
{
    MCTSNode root(state);
    // 选择阶段
    MCTSNode* node = &root;
    while (!node->is_leaf())
        node = node->select(c_puct);

    // 扩展与评估
    if (!node->state->is_terminal())
    {
        // 调用Python获取策略和价值
        py::tuple result = policy_value_fn(node->state);
        auto probs = result[0].cast<std::vector<float>>();
        float value = result[1].cast<float>();

        node->expand(probs);
        node->update(value);
    }
    else
    {
        if (node->state->get_result() == node->state->get_current_player())
            node->update(1.0f);
        else
            node->update(-1.0f);
    }
}
