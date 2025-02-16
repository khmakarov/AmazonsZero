# test_demo.py
import numpy as np
import libamazons


class RandomModel:
    """用于测试的随机策略模型"""

    def predict(self, state):
        legal_actions = state.get_legal_actions()
        probs = np.random.rand(len(legal_actions))
        probs /= probs.sum()
        value = np.random.uniform(-1, 1)
        return probs.tolist(), float(value)


def policy_value_wrapper(model):
    """将模型预测转换为C++需要的格式"""

    def fn(game_state):
        # 将C++状态转换为numpy
        legal_actions = game_state.get_legal_actions()

        # 随机模型生成示例
        probs, value = model.predict(game_state)
        return (probs, value)

    return fn


def main():
    # 初始化组件
    game = libamazons.GameCore()
    model = RandomModel()
    mcts = libamazons.MCTSEngine(n_simulations=100, c_puct=1.25)

    # 运行完整流程
    for step in range(3):
        print(f"Step {step+1}:")
        # MCTS搜索
        action_probs = mcts.run(game, policy_value_wrapper(model))

        # 选择动作
        legal_actions = game.get_legal_actions()
        chosen_idx = np.argmax(action_probs)
        chosen_action = legal_actions[chosen_idx]

        # 执行动作
        game.step(chosen_action)
        print(f"Chosen action: {chosen_action}")

        if game.is_terminal():
            print(f"Game Over! Winner: {game.get_result()}")
            break


if __name__ == "__main__":
    main()
