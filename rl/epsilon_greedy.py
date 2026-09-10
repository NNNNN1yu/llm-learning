"""
学习强化学习的 Q-Learning 与 SARSA 算法，
演示 On-Policy 与 Off-Policy 的区别。
"""

import numpy as np


ALPHA = 0.5
GAMMA = 0.9
EPSILON = 0.1
NUM_EPISODES = 1000


def epsilon_greedy(state, q_table, epsilon, rng):
    """使用 epsilon-greedy 策略选择动作。"""
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[1]))

    # 随机处理多个动作 Q 值相同的情况，
    # 避免 np.argmax 总是偏向第一个动作。
    max_q = np.max(q_table[state])
    best_actions = np.flatnonzero(q_table[state] == max_q)
    return int(rng.choice(best_actions))
