from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from panda_gym.envs.core import RobotTaskEnv
from stable_baselines3 import PPO, SAC, TD3

try:
    from sb3_contrib import TQC
except ImportError:
    TQC = None

from case0_rl_train import JointControlPanda as Case0Robot
from case0_rl_train import PandaReachCase0Task
from case1_rl_train import JointControlPanda as Case1Robot
from case1_rl_train import PandaReachWallCase1Task
from case2_rl_train import JointControlPanda as Case2Robot
from case2_rl_train import PandaReachWallCase2Task


DEFAULT_MAX_STEPS = 300


def get_model_class(algo_name: str):
    """Return the Stable-Baselines model class for the selected algorithm."""
    algo = algo_name.lower()
    if algo == "ppo":
        return PPO
    if algo == "sac":
        return SAC
    if algo == "td3":
        return TD3
    if algo == "tqc":
        if TQC is None:
            raise ImportError(
                "The selected algorithm is TQC, but sb3-contrib is not installed. "
                "Install it with: python -m pip install sb3-contrib"
            )
        return TQC
    raise ValueError("Supported algorithms: ppo, sac, td3, tqc")


def get_case_components(case_id: int):
    """Map the selected case ID to the matching robot and task classes."""
    if case_id == 0:
        return Case0Robot, PandaReachCase0Task
    if case_id == 1:
        return Case1Robot, PandaReachWallCase1Task
    if case_id == 2:
        return Case2Robot, PandaReachWallCase2Task
    raise ValueError("Supported case IDs: 0, 1, 2")


def build_env(case_id: int, render: bool) -> gym.Env:
    """Build a single PandaGym environment for the requested case."""
    from panda_gym.pybullet import PyBullet

    robot_cls, task_cls = get_case_components(case_id)
    sim = PyBullet(render_mode="human") if render else PyBullet()
    robot = robot_cls(sim)
    task = task_cls(sim, get_ee_position=robot.get_ee_position, reward_type="dense")
    env = RobotTaskEnv(robot, task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=DEFAULT_MAX_STEPS)
    return env


def default_model_path(case_id: int, algo_name: str, model_dir: str | Path) -> Path:
    """Build the default path for the best saved policy."""
    return Path(model_dir) / f"case_{case_id}_{algo_name}_best_policy.zip"


def extract_success(info: dict) -> bool:
    """Extract the success flag from PandaGym info in a tolerant way."""
    value = info.get("is_success", False)
    if isinstance(value, np.ndarray):
        return bool(np.asarray(value).astype(float).item() >= 1.0)
    return bool(float(value) >= 1.0)


def get_goal_position(env: gym.Env) -> np.ndarray:
    """Read the fixed goal position from the current environment observation."""
    if hasattr(env.unwrapped, "task") and hasattr(env.unwrapped.task, "goal"):
        return np.asarray(env.unwrapped.task.goal, dtype=float).copy()
    raise RuntimeError("Could not access task goal from the environment")


def _get_case_scene(case_id: int) -> dict:
    """Return the workspace geometry used for 3D overlay visualization."""
    if case_id == 0:
        return {
            "goal": np.array([0.4, 0.0, 0.1], dtype=float),
            "walls": [],
            "xlim": (0.0, 0.65),
            "ylim": (-0.45, 0.45),
            "zlim": (0.0, 0.55),
        }
    if case_id == 1:
        return {
            "goal": np.array([0.4, 0.0, 0.1], dtype=float),
            "walls": [
                {
                    "x_min": 0.20 - 0.01 / 2.0,
                    "x_max": 0.20 + 0.01 / 2.0,
                    "y_min": -0.2,
                    "y_max": 0.2,
                    "z_min": 0.0,
                    "z_max": 0.4,
                }
            ],
            "xlim": (0.0, 0.65),
            "ylim": (-0.45, 0.45),
            "zlim": (0.0, 0.55),
        }
    if case_id == 2:
        return {
            "goal": np.array([0.4, 0.0, 0.05], dtype=float),
            "walls": [
                {
                    "x_min": 0.35 - 0.02 / 2.0,
                    "x_max": 0.35 + 0.02 / 2.0,
                    "y_min": -1.5 / 2.0,
                    "y_max": 1.5 / 2.0,
                    "z_min": 0.30,
                    "z_max": 0.80,
                }
            ],
            "xlim": (0.0, 0.65),
            "ylim": (-0.8, 0.8),
            "zlim": (0.0, 0.9),
        }
    raise ValueError(f"Unsupported case_id: {case_id}")


def _make_box_faces(box: dict) -> list[list[list[float]]]:
    """Convert a box definition into polygon faces for Matplotlib 3D drawing."""
    x0, x1 = box["x_min"], box["x_max"]
    y0, y1 = box["y_min"], box["y_max"]
    z0, z1 = box["z_min"], box["z_max"]

    v000 = [x0, y0, z0]
    v100 = [x1, y0, z0]
    v110 = [x1, y1, z0]
    v010 = [x0, y1, z0]
    v001 = [x0, y0, z1]
    v101 = [x1, y0, z1]
    v111 = [x1, y1, z1]
    v011 = [x0, y1, z1]

    return [
        [v000, v100, v110, v010],
        [v001, v101, v111, v011],
        [v000, v100, v101, v001],
        [v010, v110, v111, v011],
        [v000, v010, v011, v001],
        [v100, v110, v111, v101],
    ]


def plot_robustness_trajectories_overlay(
    trajectories: List[np.ndarray],
    case_id: int,
    algo_name: str,
    output_path: str | Path | None = None,
    show: bool = True,
):
    """Plot many end-effector trajectories in one 3D figure without a legend."""
    if not trajectories:
        raise ValueError("trajectories cannot be empty")

    scene = _get_case_scene(case_id)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    for wall in scene["walls"]:
        wall_poly = Poly3DCollection(
            _make_box_faces(wall),
            alpha=0.25,
            facecolor="salmon",
            edgecolor="none",
        )
        ax.add_collection3d(wall_poly)

    for traj in trajectories:
        ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], linewidth=1.2, alpha=0.35)

    start = trajectories[0][0]
    goal = scene["goal"]
    ax.scatter(start[0], start[1], start[2], color="green", s=45)
    ax.scatter(goal[0], goal[1], goal[2], color="magenta", s=70)

    ax.set_title(f"Algorithm: {algo_name.upper()} | Runs: {len(trajectories)}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(*scene["xlim"])
    ax.set_ylim(*scene["ylim"])
    ax.set_zlim(*scene["zlim"])

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")

    if show:
        plt.show()

    return fig


def run_one_episode(env: gym.Env, model, deterministic: bool, max_steps: int) -> dict:
    """Run one evaluation episode and return scalar metrics plus the EE trajectory."""
    obs, _ = env.reset()
    goal_pos = get_goal_position(env)

    ee_positions = [env.unwrapped.robot.get_ee_position().copy()]
    total_reward = 0.0
    success = False
    terminated = False
    truncated = False
    final_distance = np.linalg.norm(ee_positions[-1] - goal_pos)

    for step in range(1, max_steps + 1):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        ee_pos = env.unwrapped.robot.get_ee_position().copy()
        ee_positions.append(ee_pos)
        final_distance = float(np.linalg.norm(ee_pos - goal_pos))

        success = extract_success(info)
        if terminated or truncated:
            break

    return {
        "success": int(success),
        "final_distance": final_distance,
        "num_steps": len(ee_positions) - 1,
        "total_reward": total_reward,
        "terminated": int(terminated),
        "truncated": int(truncated),
        "ee_trajectory": np.asarray(ee_positions, dtype=float),
    }


def save_summary_csv(output_path: str | Path, rows: List[dict]) -> None:
    """Save per-episode robustness metrics into one summary CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "episode_id",
                "success",
                "final_distance",
                "num_steps",
                "total_reward",
                "terminated",
                "truncated",
            ]
        )
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                [
                    idx,
                    row["success"],
                    round(float(row["final_distance"]), 6),
                    row["num_steps"],
                    round(float(row["total_reward"]), 6),
                    row["terminated"],
                    row["truncated"],
                ]
            )


def print_summary(rows: List[dict]) -> None:
    """Print aggregate robustness statistics to the console."""
    successes = np.array([row["success"] for row in rows], dtype=float)
    final_distances = np.array([row["final_distance"] for row in rows], dtype=float)
    num_steps = np.array([row["num_steps"] for row in rows], dtype=float)
    rewards = np.array([row["total_reward"] for row in rows], dtype=float)

    print("\n=== Robustness Summary ===")
    print(f"Runs: {len(rows)}")
    print(f"Successful runs: {int(successes.sum())}/{len(rows)}")
    print(f"Success rate [%]: {100.0 * successes.mean():.2f}")
    print(f"Mean final distance [m]: {final_distances.mean():.6f}")
    print(f"Mean total reward: {rewards.mean():.6f}")
    print(f"Mean number of steps: {num_steps.mean():.2f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the robustness of one trained RL policy over N runs using stochastic inference "
            "(deterministic=False)."
        )
    )
    parser.add_argument("--case-id", type=int, required=True, choices=[0, 1, 2])
    parser.add_argument("--algo", type=str, required=True, choices=["ppo", "sac", "td3", "tqc"])
    parser.add_argument("--model", type=str, default=None, help="Path to the trained .zip policy")
    parser.add_argument(
        "--model-dir",
        type=str,
        default=".",
        help="Directory used when --model is not given. The script will look for case_<id>_<algo>_best_policy.zip",
    )
    parser.add_argument("--num-runs", type=int, default=100, help="Number of robustness evaluation runs")
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS, help="Maximum steps per episode")
    parser.add_argument("--render", action="store_true", help="Show PyBullet human render mode during execution")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where the summary CSV and optional overlay figure will be saved",
    )
    parser.add_argument(
        "--plot-overlay",
        action="store_true",
        help="Create a 3D overlay figure with all end-effector trajectories",
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model is not None else default_model_path(args.case_id, args.algo, args.model_dir)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = build_env(case_id=args.case_id, render=args.render)
    model_cls = get_model_class(args.algo)
    model = model_cls.load(str(model_path), env=env)

    if args.algo.lower() in {"td3"}:
        print(
            "Warning: TD3 uses a deterministic policy network. Even with deterministic=False, "
            "runs may still be effectively identical if the environment is deterministic."
        )

    rows: List[dict] = []
    for episode_idx in range(1, args.num_runs + 1):
        result = run_one_episode(env, model, deterministic=False, max_steps=args.max_steps)
        rows.append(result)
        print(
            f"Run {episode_idx:03d}/{args.num_runs} | "
            f"success={result['success']} | "
            f"final_distance={result['final_distance']:.6f} m | "
            f"steps={result['num_steps']} | "
            f"reward={result['total_reward']:.6f}"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_output = output_dir / f"case_{args.case_id}_{args.algo}_robustness_summary.csv"
    save_summary_csv(summary_output, rows)
    print(f"\nSaved robustness summary -> {summary_output}")

    if args.plot_overlay:
        overlay_output = output_dir / f"case_{args.case_id}_{args.algo}_robustness_overlay.png"
        trajectories = [row["ee_trajectory"] for row in rows]
        plot_robustness_trajectories_overlay(
            trajectories=trajectories,
            case_id=args.case_id,
            algo_name=args.algo,
            output_path=overlay_output,
            show=True,
        )
        print(f"Saved robustness overlay -> {overlay_output}")

    print_summary(rows)
    env.close()


if __name__ == "__main__":
    main()

