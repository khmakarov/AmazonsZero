# mcts_wrapper.py
import libamazons


class MCTS:

    def __init__(self, model, c_puct=1.25):
        self.searcher = libamazons.MCTSSearcher(n_threads=4)
        self.model = model
        self.c_puct = c_puct

    def get_action_probs(self, state, simulations=800):
        # 将状态转换为神经网络输入格式
        state_tensor = preprocess(state)

        # 调用C++搜索核心
        return self.searcher.run(state=state,
                                 simulations=simulations,
                                 policy_fn=lambda s: self.model.predict(s))
