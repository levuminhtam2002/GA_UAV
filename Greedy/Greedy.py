import numpy as np
import math
from UAV_env import UAVEnv
import copy
import matplotlib.pyplot as plt

def greedy_algorithm(env):
    env.reset()
    total_reward = 0
    uav_positions = []
    actions_log = []
    for _ in range(env.slot_num):
        best_action = None
        best_reward = float('inf')
        for ue_id in range(env.M):
            for offloading_ratio in np.linspace(0, 1, 11):
                action = np.array([ue_id / env.M, np.random.uniform(), np.random.uniform(), offloading_ratio])
                state, reward, is_terminal, _, _, _ = env.step(action)
                if reward < best_reward:  # Minimize reward (delay)
                    best_reward = reward
                    best_action = action
        state, reward, is_terminal, _, _, _ = env.step(best_action)
        total_reward += reward
        uav_positions.append(env.loc_uav.copy())
        actions_log.append(best_action)
        if is_terminal:
            break
    return total_reward, uav_positions, actions_log


def find_optimal_solution(env, num_runs=100):
    best_total_reward = float('inf')
    best_uav_positions = []
    best_actions_log = []
    for _ in range(num_runs):
        total_reward, uav_positions, actions_log = greedy_algorithm(env)
        if total_reward < best_total_reward:
            best_total_reward = total_reward
            best_uav_positions = uav_positions
            best_actions_log = actions_log

    with open("optimal_solution.txt", "w") as file:
        for i, (pos, action) in enumerate(zip(best_uav_positions, best_actions_log)):
            ue_id = int(action[0] * env.M)
            offloading_ratio = action[3]
            file.write(f"Step {i + 1}:\n")
            file.write(f"  UAV Position: {pos}\n")
            file.write(f"  UE Served: UE {ue_id + 1}\n")
            file.write(f"  Offloading Ratio: {offloading_ratio:.2f}\n")
            file.write(f"  Delay (Reward): {-best_total_reward:.2f}\n")
            file.write("\n")
    return best_total_reward, best_uav_positions, best_actions_log

env = UAVEnv()
find_optimal_solution(env, num_runs=100)