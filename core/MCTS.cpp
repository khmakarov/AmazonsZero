#include "mcts.h"
MCTSNode *MCTSNode::select(double c_puct)
{
    MCTSNode *best_child = nullptr;
    double best_uct = -INFINITY;

    for (auto &[action_hash, child] : children)
    {
        double uct = child->value_sum / (child->visit_count + 1e-5) +
                     c_puct * child->prior *
                         sqrt(visit_count) / (child->visit_count + 1);
        if (uct > best_uct)
        {
            best_uct = uct;
            best_child = child.get();
        }
    }
    return best_child;
}
