"""Generate pre-computed worlds and reward configurations.

Produces:
  saved/worlds.npz        -- 1000 world matrices
  saved/rewards_fixed.npz -- fixed reward assignments (no noise)
  saved/rewards_info.npz  -- reward assignments with stochastic noise
"""

import numpy as np
import os

from utils.world import VillageWorld
from utils.helper_functions import assign_fixed_rewards, sample_reward_offset

n_simulations = 1000
n_episodes = 120
target_var = 100

rng = np.random.default_rng(5)

os.makedirs('saved', exist_ok=True)

# --- Stage A: Generate 1000 random worlds ---
print("Generating worlds...")
worlds = []
for s in range(n_simulations):
    env = VillageWorld(rng=rng)
    worlds.append(env.get_world_matrix())

np.savez_compressed('saved/worlds.npz', *worlds)
print(f"  saved/worlds.npz  ({len(worlds)} worlds)")

# Reload so downstream stages use exactly the saved arrays
worlds_load = np.load('saved/worlds.npz')
worlds_saved = [worlds_load[f'arr_{i}'] for i in range(len(worlds_load.files))]

# --- Stage B: Assign fixed rewards per simulation ---
print("Generating fixed rewards...")
save_rewards_info = []
for s in range(n_simulations):
    env = VillageWorld(worlds_saved[s], rng=rng)
    reward_values = assign_fixed_rewards(env, rng)
    rewards_episode = [reward_values.copy() for _ in range(n_episodes)]
    save_rewards_info.append(rewards_episode)

np.savez_compressed('saved/rewards_fixed.npz', *save_rewards_info)
print(f"  saved/rewards_fixed.npz  ({n_simulations} sims x {n_episodes} episodes)")

# Reload fixed rewards for noise stage
rewards_fixed_load = np.load('saved/rewards_fixed.npz')
rewards_fixed = [rewards_fixed_load[f'arr_{i}'] for i in range(len(rewards_fixed_load.files))]

# --- Stage C: Add stochastic noise to rewards ---
print("Adding noise to rewards...")
n_rewards = rewards_fixed[0][0].shape[0]
offsets = np.array([
    sample_reward_offset(target_var)
    for _ in range(n_simulations * n_episodes * n_rewards)
])

save_rewards_info_noise = []
for sim in range(n_simulations):
    sim_rewards_noisy = []
    for episode in range(n_episodes):
        episode_rewards = rewards_fixed[sim][episode].copy()
        for idx in range(episode_rewards.shape[0]):
            if episode_rewards[idx, 1] != 0:
                offset_idx = sim * n_episodes * n_rewards + episode * n_rewards + idx
                episode_rewards[idx, 1] += offsets[offset_idx]
                episode_rewards[idx, 1] = max(episode_rewards[idx, 1], 0)
        sim_rewards_noisy.append(episode_rewards)
    save_rewards_info_noise.append(sim_rewards_noisy)

np.savez_compressed('saved/rewards_info.npz', *save_rewards_info_noise)
print(f"  saved/rewards_info.npz  ({n_simulations} sims x {n_episodes} episodes, var={target_var})")

print("Done.")
