"""
Compare the three teacher conditions across MULTIPLE worlds to see whether the
pedagogical objectives systematically fail (vs. just unlucky on one world).

Conditions (same as plot_pedagogical_trajectories.py):
    - passive (greedy)  : objective="cumulative_reward" (non-pedagogical baseline)
    - q_mismatch        : pedagogical objective 2
    - action_gap        : pedagogical objective 3

For each of N_WORLDS worlds we run all three conditions on the SAME world & seed,
then aggregate the single learner's reward and steps-to-reward across worlds
(mean +/- standard error) for every social episode (teaching + solo test).

Run from the repository root:
    python -m utils.compare_pedagogical_worlds
"""

import io
import json
import os
from contextlib import redirect_stdout

import numpy as np
import matplotlib.pyplot as plt

from models.mb_pedagogical_expert import mb_pedagogical_expert
from models.mf_valueshaping import MFValueShapingAgent
from utils.world import VillageWorld
from utils.plot_functions import plot_with_se


# ----------------------------- Configuration ----------------------------- #
exp = "baseline"
BASE_SEED = 5
N_WORLDS = 10                      # number of worlds to average over
n_train_episodes = 100
n_teach_episodes = 10              # teacher demonstrates, learner value-shapes
n_test_episodes = 10              # teacher leaves, learner runs solo
n_social = n_teach_episodes + n_test_episodes
n_episodes_total = n_train_episodes + n_social
max_steps = 40
n_learner = 1

CONDITIONS = {
    "cumulative_reward": ("passive (greedy)", "tab:gray"),
    "q_mismatch":        ("q_mismatch",       "tab:purple"),
    "action_gap":        ("action_gap",       "tab:orange"),
}


# ------------------------------- Load data ------------------------------- #
with open("saved/opti_results/mbased_expert.json", "r") as f:
    teacher_params = json.load(f)["opti_params"]
with open("saved/opti_results/mfree_vshaping.json", "r") as f:
    learner_params = json.load(f)["opti_params"]

worlds_load = np.load("saved/worlds.npz")
worlds_saved = [worlds_load[f"arr_{i}"] for i in range(len(worlds_load.files))]
rewards_load = np.load("saved/rewards_info.npz")
rewards_saved = [rewards_load[f"arr_{i}"] for i in range(len(rewards_load.files))]


def run(objective, world_idx):
    """Run one condition on one world. Same per-world seed across conditions."""
    world_matrix = worlds_saved[world_idx]
    rewards_info = rewards_saved[world_idx]
    rng = np.random.default_rng(BASE_SEED + world_idx)
    env = VillageWorld(world_matrix, rng)
    with redirect_stdout(io.StringIO()):  # silence the learner's per-episode print
        out = mb_pedagogical_expert(
            teacher_params, env, world_matrix, rewards_info, max_steps,
            n_episodes_total, n_teach_episodes, rng, optimization=False, exp=exp,
            learner_function=MFValueShapingAgent, learner_params=learner_params,
            n_learner=n_learner, n_test_episodes=n_test_episodes, objective=objective)
    reward_sums_epi = out[1]
    learner_log = out[11][0]
    return reward_sums_epi, learner_log


# Storage: (n_worlds, n_social) per condition
learner_reward = {o: np.zeros((N_WORLDS, n_social)) for o in CONDITIONS}
learner_steps = {o: np.zeros((N_WORLDS, n_social)) for o in CONDITIONS}
teacher_teach_reward = {o: np.zeros(N_WORLDS) for o in CONDITIONS}  # mean over teaching eps

print(f"Running {len(CONDITIONS)} conditions x {N_WORLDS} worlds ...")
for obj in CONDITIONS:
    for w in range(N_WORLDS):
        reward_sums_epi, log = run(obj, w)
        learner_reward[obj][w] = log["reward_sum_episode"]
        learner_steps[obj][w] = log["steps_to_reward"]
        teacher_teach_reward[obj][w] = np.mean(
            reward_sums_epi[n_train_episodes:n_train_episodes + n_teach_episodes])
    print(f"  done: {CONDITIONS[obj][0]}")


# ------------------------------- Summary --------------------------------- #
teach_sl = slice(0, n_teach_episodes)
test_sl = slice(n_teach_episodes, n_social)


def reached_rate(steps_arr, sl):
    """Fraction of episodes where the learner reached the reward (< max_steps)."""
    return np.mean(steps_arr[:, sl] < max_steps)


print("\n=== Aggregated over "
      f"{N_WORLDS} worlds (mean) ===")
header = f"{'condition':18s} | {'teacher teach R':>15s} | "\
         f"{'learner teach R':>15s} {'test R':>8s} | "\
         f"{'reach% teach':>12s} {'reach% test':>11s}"
print(header)
print("-" * len(header))
for obj, (label, _) in CONDITIONS.items():
    lt = np.mean(learner_reward[obj][:, teach_sl])
    lte = np.mean(learner_reward[obj][:, test_sl])
    rt = reached_rate(learner_steps[obj], teach_sl) * 100
    rte = reached_rate(learner_steps[obj], test_sl) * 100
    tt = np.mean(teacher_teach_reward[obj])
    print(f"{label:18s} | {tt:15.1f} | {lt:15.1f} {lte:8.1f} | "
          f"{rt:11.0f}% {rte:10.0f}%")


# ------------------------------- Plots ----------------------------------- #
social_x = np.arange(1, n_social + 1)
boundary = n_teach_episodes + 0.5

fig, (ax_r, ax_s) = plt.subplots(1, 2, figsize=(14, 5))
for obj, (label, color) in CONDITIONS.items():
    plot_with_se(ax_r, social_x, learner_reward[obj], color, label)
    plot_with_se(ax_s, social_x, learner_steps[obj], color, label)

for ax, title, ylab in [(ax_r, "Learner reward sum per episode", "Reward sum"),
                        (ax_s, "Learner steps to reward per episode", "Steps to reward")]:
    ax.axvline(boundary, linestyle="--", color="gray")
    ymax = ax.get_ylim()[1]
    ax.text(n_teach_episodes / 2 + 0.5, ymax, "teaching", ha="center", va="top",
            fontsize=10, color="gray")
    ax.text(n_teach_episodes + n_test_episodes / 2 + 0.5, ymax, "test (solo)",
            ha="center", va="top", fontsize=10, color="gray")
    ax.set_title(title)
    ax.set_xlabel("Social episode")
    ax.set_ylabel(ylab)
    ax.legend()
fig.suptitle(f"Learner performance averaged over {N_WORLDS} worlds (mean +/- SE)",
             fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.96])

save_dir = os.path.join("saved", "figures", "pedagogical_demo")
os.makedirs(save_dir, exist_ok=True)
out_path = os.path.join(save_dir, f"learner_performance_{N_WORLDS}worlds.png")
fig.savefig(out_path, bbox_inches="tight", dpi=150)
print(f"\nSaved -> {out_path}")

plt.show()
