"""
Visualize the model-based pedagogical expert(s) and their learner on a SINGLE world.

Compares three teacher conditions on the same world & seed:
    - passive (greedy)  : teacher demonstrates its own reward-greedy action
                          (objective="cumulative_reward") -- the non-pedagogical
                          "passive expert" baseline.
    - q_mismatch        : pedagogical objective 2
    - action_gap        : pedagogical objective 3

Produces:
  1) Teacher performance (reward sum per episode) for all three conditions.
  2) Learner performance (reward sum and steps-to-reward) across BOTH the
     teaching phase and the solo test phase, for all three conditions.
  3) Teacher (blue) vs learner (red) trajectories for the LAST N_LAST teaching
     episodes, as a grid of [condition x episode].

Phases (per run):
    episodes   0 .. 99   : asocial teacher training (learner inactive)
    episodes 100 .. 109   : teaching  (teacher demonstrates, learner value-shapes)
    episodes 110 .. 119   : test      (teacher leaves, learner runs solo)

Run from the repository root:
    python -m utils.plot_pedagogical_trajectories
"""

import json
import os

import numpy as np
import matplotlib.pyplot as plt

from models.mb_pedagogical_expert import mb_pedagogical_expert
from models.mf_valueshaping import MFValueShapingAgent
from utils.world import VillageWorld
from utils.plot_functions import state_to_xy, add_boundaries


# ----------------------------- Configuration ----------------------------- #
exp = "baseline"
SEED = 5
SIM_INDEX = 0                      # which saved world to use as the example
n_train_episodes = 100             # asocial teacher training
n_teach_episodes = 10              # teacher demonstrates, learner value-shapes
n_test_episodes = 10               # teacher leaves, learner runs solo
n_episodes_total = n_train_episodes + n_teach_episodes + n_test_episodes
max_steps = 40
n_learner = 1                      # a single learner learns with the teacher

# Teacher conditions: objective name -> display label & color
CONDITIONS = {
    "cumulative_reward": ("passive (greedy)", "tab:gray"),
    "q_mismatch":        ("q_mismatch",       "tab:purple"),
    "action_gap":        ("action_gap",       "tab:orange"),
}

N_LAST = 5                         # number of (last) teaching episodes to draw
# 0-indexed social episodes of the last N_LAST teaching episodes
TRAJ_EPISODES = list(range(n_teach_episodes - N_LAST, n_teach_episodes))


# ------------------------------- Load data ------------------------------- #
with open("saved/opti_results/mbased_expert.json", "r") as f:
    teacher_params = json.load(f)["opti_params"]
with open("saved/opti_results/mfree_vshaping.json", "r") as f:
    learner_params = json.load(f)["opti_params"]

worlds_load = np.load("saved/worlds.npz")
worlds_saved = [worlds_load[f"arr_{i}"] for i in range(len(worlds_load.files))]

rewards_load = np.load("saved/rewards_info.npz")
rewards_saved = [rewards_load[f"arr_{i}"] for i in range(len(rewards_load.files))]

world_matrix = worlds_saved[SIM_INDEX]
rewards_info = rewards_saved[SIM_INDEX]
grid_size = world_matrix.shape[0]


def run_objective(objective):
    """Run the pedagogical expert once for a given objective on the example world.

    A fresh rng (same seed) and a fresh env (same world matrix) are used so that
    the asocial training phase is identical across conditions and the only thing
    that differs is the teaching strategy.
    """
    rng = np.random.default_rng(SEED)
    env = VillageWorld(world_matrix, rng)

    out = mb_pedagogical_expert(
        teacher_params,
        env,
        world_matrix,
        rewards_info,
        max_steps,
        n_episodes_total,
        n_teach_episodes,
        rng,
        optimization=False,
        exp=exp,
        learner_function=MFValueShapingAgent,
        learner_params=learner_params,
        n_learner=n_learner,
        n_test_episodes=n_test_episodes,
        objective=objective,
    )
    (value, reward_sums_epi, state_mat, action_mat, steps_to_reward,
     tm_final, model_r, value_epi, tm_epi, reward_per_step,
     teacher_predictions, learner_logs) = out

    return {
        "env": env,
        "reward_sums_epi": reward_sums_epi,     # teacher, per episode (len n_episodes_total)
        "state_mat": state_mat,                 # teacher trajectory, per episode
        "learner_log": learner_logs[0],         # the single learner's full log
    }


results = {obj: run_objective(obj) for obj in CONDITIONS}


# --------------------------- Trajectory drawing -------------------------- #
def draw_world(ax, env, world):
    ax.set_xlim([0, grid_size])
    ax.set_ylim([0, grid_size])
    ax.set_aspect("equal")
    for i in range(grid_size + 1):
        ax.axhline(i, color="lightgray", linewidth=0.6)
        ax.axvline(i, color="lightgray", linewidth=0.6)
    ax.axis("off")

    for s in env.reward_states:
        y, x = state_to_xy(world, s)
        ax.add_patch(plt.Rectangle((x, grid_size - y - 1), 1, 1, color="#cfe8cf"))
        ax.text(x + 0.5, grid_size - y - 0.5, "R", va="center", ha="center",
                color="#2e7d32", fontsize=8, fontweight="bold")

    try:
        boundaries = env.get_all_boundaries()
        if boundaries:
            add_boundaries(ax, world, boundaries, grid_size)
    except Exception:
        pass


def draw_path(ax, world, path, cmap_name, label):
    """Draw a trajectory as a color-graded line; mark start (o) and end (X)."""
    path = np.asarray(path, dtype=float)
    path = path[~np.isnan(path)].astype(int)
    if len(path) == 0:
        return

    cmap = plt.get_cmap(cmap_name)
    n_seg = max(len(path) - 1, 1)
    seg_colors = cmap(np.linspace(0.35, 0.95, n_seg))

    for i in range(len(path) - 1):
        y0, x0 = state_to_xy(world, path[i])
        y1, x1 = state_to_xy(world, path[i + 1])
        ax.plot([x0 + 0.5, x1 + 0.5],
                [grid_size - y0 - 0.5, grid_size - y1 - 0.5],
                color=seg_colors[i], linewidth=3, solid_capstyle="round", zorder=3)

    y_s, x_s = state_to_xy(world, path[0])
    ax.scatter(x_s + 0.5, grid_size - y_s - 0.5, color=cmap(0.25), s=80,
               marker="o", edgecolor="k", zorder=5, label=f"{label} start")
    y_e, x_e = state_to_xy(world, path[-1])
    ax.scatter(x_e + 0.5, grid_size - y_e - 0.5, color=cmap(0.95), s=90,
               marker="X", edgecolor="k", zorder=5, label=f"{label} end")


# Grid: rows = teacher condition, columns = last N_LAST teaching episodes
fig_traj, axes = plt.subplots(len(CONDITIONS), N_LAST,
                              figsize=(3.2 * N_LAST, 3.4 * len(CONDITIONS)))
axes = np.atleast_2d(axes)

for r, obj in enumerate(CONDITIONS):
    label, _ = CONDITIONS[obj]
    res = results[obj]
    for c, se in enumerate(TRAJ_EPISODES):
        ax = axes[r, c]
        teacher_path = res["state_mat"][n_train_episodes + se]
        learner_path = res["learner_log"]["states"][se]
        l_steps = int(res["learner_log"]["steps_to_reward"][se])

        draw_world(ax, res["env"], world_matrix)
        draw_path(ax, world_matrix, teacher_path, "Blues", "Teacher")
        draw_path(ax, world_matrix, learner_path, "Reds", "Learner")

        if r == 0:
            ax.set_title(f"teaching ep {se + 1}/{n_teach_episodes}", fontsize=11)
        if c == 0:
            ax.text(-0.08, 0.5, label, transform=ax.transAxes, rotation=90,
                    va="center", ha="right", fontsize=12, fontweight="bold")
        ax.text(0.5, -0.04, f"learner: {l_steps} steps", transform=ax.transAxes,
                va="top", ha="center", fontsize=8, color="dimgray")

fig_traj.suptitle("Teacher (blue) vs learner (red) trajectories — last "
                  f"{N_LAST} teaching episodes (same world & seed)", fontsize=14)
fig_traj.tight_layout(rect=[0.02, 0, 1, 0.97])


# ------------------------- Teacher performance --------------------------- #
fig_teacher, ax_t = plt.subplots(figsize=(8, 5))
episodes_x = np.arange(1, n_episodes_total + 1)
teach_start = n_train_episodes + 1
test_start = n_train_episodes + n_teach_episodes + 1

for obj, (label, color) in CONDITIONS.items():
    ax_t.plot(episodes_x, results[obj]["reward_sums_epi"],
              color=color, label=label, linewidth=1.6)
ax_t.axvline(teach_start - 0.5, linestyle="--", color="gray")
ax_t.axvline(test_start - 0.5, linestyle="--", color="gray")
ax_t.set_title("Teacher: reward sum per episode")
ax_t.set_xlabel("Episode")
ax_t.set_ylabel("Reward sum")
ax_t.legend()
fig_teacher.tight_layout()


# ------------------------- Learner performance --------------------------- #
# Social-episode axis: 1..n_teach = teaching, n_teach+1..n_social = test
n_social = n_teach_episodes + n_test_episodes
social_x = np.arange(1, n_social + 1)
boundary = n_teach_episodes + 0.5

fig_learner, (ax_lr, ax_ls) = plt.subplots(1, 2, figsize=(14, 5))
for obj, (label, color) in CONDITIONS.items():
    log = results[obj]["learner_log"]
    ax_lr.plot(social_x, log["reward_sum_episode"], color=color, marker="o",
               markersize=4, label=label, linewidth=1.6)
    ax_ls.plot(social_x, log["steps_to_reward"], color=color, marker="o",
               markersize=4, label=label, linewidth=1.6)

for ax, title, ylab in [(ax_lr, "Learner: reward sum per episode", "Reward sum"),
                        (ax_ls, "Learner: steps to reward per episode", "Steps to reward")]:
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
fig_learner.tight_layout()


# ------------------------------- Save / show ----------------------------- #
save_dir = os.path.join("saved", "figures", "pedagogical_demo")
os.makedirs(save_dir, exist_ok=True)
traj_path = os.path.join(save_dir, f"trajectories_sim{SIM_INDEX}_last{N_LAST}.png")
teacher_path_fig = os.path.join(save_dir, f"teacher_performance_sim{SIM_INDEX}.png")
learner_path_fig = os.path.join(save_dir, f"learner_performance_sim{SIM_INDEX}.png")
fig_traj.savefig(traj_path, bbox_inches="tight", dpi=150)
fig_teacher.savefig(teacher_path_fig, bbox_inches="tight", dpi=150)
fig_learner.savefig(learner_path_fig, bbox_inches="tight", dpi=150)

print(f"Saved trajectory figure  -> {traj_path}")
print(f"Saved teacher figure      -> {teacher_path_fig}")
print(f"Saved learner figure      -> {learner_path_fig}")

plt.show()
