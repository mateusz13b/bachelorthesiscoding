from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import gymnasium as gym
import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from stable_baselines3 import PPO, SAC, TD3

try:
    from sb3_contrib import TQC
except ImportError:  # pragma: no cover
    TQC = None

# Required environment imports from the user project.
from case0_rl_train import JointControlPanda as Case0Robot
from case0_rl_train import PandaReachCase0Task
from case1_rl_train import JointControlPanda as Case1Robot
from case1_rl_train import PandaReachWallCase1Task
from case2_rl_train import JointControlPanda as Case2Robot
from case2_rl_train import PandaReachWallCase2Task


SUPPORTED_ALGOS = ("ppo", "sac", "td3", "tqc")
SUPPORTED_CASES = (0, 1, 2)
DEFAULT_MAX_EPISODE_STEPS = 300
DEFAULT_REWARD_TYPE = "dense"


def get_model_class(algo_name: str):
    """Return the Stable-Baselines3 class for the selected algorithm."""
    algo_key = algo_name.lower()
    if algo_key == "ppo":
        return PPO
    if algo_key == "sac":
        return SAC
    if algo_key == "td3":
        return TD3
    if algo_key == "tqc":
        if TQC is None:
            raise ImportError(
                "TQC requires sb3-contrib. Install it with: python -m pip install sb3-contrib"
            )
        return TQC
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def get_train_script_name(case_id: int) -> str:
    """Map the case ID to the expected training script filename."""
    mapping = {
        0: "case0_rl_train.py",
        1: "case1_rl_train.py",
        2: "case2_rl_train.py",
    }
    if case_id not in mapping:
        raise ValueError(f"Unsupported case_id: {case_id}")
    return mapping[case_id]


def make_single_env(
    case_id: int,
    render: bool = False,
    reward_type: str = DEFAULT_REWARD_TYPE,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
):
    """Create one evaluation environment for the selected case."""
    from panda_gym.pybullet import PyBullet

    if case_id == 0:
        robot_cls = Case0Robot
        task_cls = PandaReachCase0Task
    elif case_id == 1:
        robot_cls = Case1Robot
        task_cls = PandaReachWallCase1Task
    elif case_id == 2:
        robot_cls = Case2Robot
        task_cls = PandaReachWallCase2Task
    else:
        raise ValueError(f"Unsupported case_id: {case_id}")

    sim = PyBullet(render_mode="human") if render else PyBullet()
    robot = robot_cls(sim)
    task = task_cls(
        sim,
        get_ee_position=robot.get_ee_position,
        reward_type=reward_type,
    )
    env = RobotTaskEnv(robot, task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env, robot


def train_one_run(
    python_executable: str,
    train_script: Path,
    case_id: int,
    algo_name: str,
    seed: int,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    output_dir: Path,
) -> Path:
    """Launch one training run through the existing per-case training script."""
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        str(train_script),
        "--algo",
        algo_name,
        "--case-id",
        str(case_id),
        "--seed",
        str(seed),
        "--total-timesteps",
        str(total_timesteps),
        "--eval-freq",
        str(eval_freq),
        "--n-eval-episodes",
        str(n_eval_episodes),
        "--output-dir",
        str(output_dir),
    ]

    print("\n=== Training run ===")
    print(" ".join(command))
    subprocess.run(command, check=True)

    best_model_path = output_dir / f"case_{case_id}_{algo_name}_best_policy.zip"
    if not best_model_path.exists():
        raise FileNotFoundError(f"Expected best policy was not found: {best_model_path}")
    return best_model_path


def run_one_episode(model, env, deterministic: bool) -> Dict[str, float | int | bool]:
    """Run one evaluation episode and collect scalar metrics."""
    obs, _ = env.reset()
    done = False
    total_reward = 0.0
    num_steps = 0
    last_info: Dict[str, object] = {}

    while not done:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        num_steps += 1
        done = bool(terminated or truncated)
        last_info = dict(info)

    ee_pos = env.unwrapped.robot.get_ee_position().copy()
    goal_pos = np.asarray(obs["desired_goal"], dtype=float)
    final_distance = float(np.linalg.norm(ee_pos - goal_pos))

    success_raw = last_info.get("is_success", 0.0)
    if isinstance(success_raw, np.ndarray):
        success_value = float(np.asarray(success_raw).reshape(-1)[0])
    else:
        success_value = float(success_raw)

    return {
        "success": int(success_value >= 1.0),
        "final_distance": final_distance,
        "num_steps": num_steps,
        "total_reward": total_reward,
    }


def evaluate_policy(
    case_id: int,
    algo_name: str,
    model_path: Path,
    n_eval_episodes: int,
    deterministic: bool,
) -> Dict[str, float | int | str]:
    """Evaluate one trained policy on a fixed number of episodes."""
    model_class = get_model_class(algo_name)
    model = model_class.load(str(model_path))

    env, _robot = make_single_env(case_id=case_id, render=False)
    episode_rows: List[Dict[str, float | int]] = []

    try:
        for episode_id in range(n_eval_episodes):
            metrics = run_one_episode(model=model, env=env, deterministic=deterministic)
            metrics["episode_id"] = episode_id
            episode_rows.append(metrics)
    finally:
        env.close()

    num_successes = int(sum(int(row["success"]) for row in episode_rows))
    success_rate = float(num_successes / n_eval_episodes) if n_eval_episodes > 0 else 0.0
    mean_final_distance = float(np.mean([row["final_distance"] for row in episode_rows]))
    mean_total_reward = float(np.mean([row["total_reward"] for row in episode_rows]))
    mean_num_steps = float(np.mean([row["num_steps"] for row in episode_rows]))

    return {
        "case_id": case_id,
        "algo": algo_name,
        "model_path": str(model_path),
        "n_eval_episodes": n_eval_episodes,
        "num_successes": num_successes,
        "success_rate": success_rate,
        "mean_final_distance": mean_final_distance,
        "mean_total_reward": mean_total_reward,
        "mean_num_steps": mean_num_steps,
        "successful_training": int(success_rate >= 1.0),
    }


def save_detailed_summary_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Save one row per trained model."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "case_id",
        "algo",
        "seed",
        "run_dir",
        "model_path",
        "n_eval_episodes",
        "num_successes",
        "success_rate",
        "mean_final_distance",
        "mean_total_reward",
        "mean_num_steps",
        "successful_training",
        "train_seconds",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def aggregate_results(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    """Aggregate detailed rows into one summary row per case and algorithm."""
    grouped: Dict[Tuple[int, str], List[Dict[str, object]]] = {}
    for row in rows:
        key = (int(row["case_id"]), str(row["algo"]))
        grouped.setdefault(key, []).append(row)

    aggregated_rows: List[Dict[str, object]] = []
    for (case_id, algo_name), group in sorted(grouped.items()):
        num_runs = len(group)
        num_successful = int(sum(int(row["successful_training"]) for row in group))
        percent_successful = 100.0 * num_successful / num_runs if num_runs > 0 else 0.0
        mean_success_rate = float(np.mean([float(row["success_rate"]) for row in group]))
        mean_final_distance = float(np.mean([float(row["mean_final_distance"]) for row in group]))
        mean_total_reward = float(np.mean([float(row["mean_total_reward"]) for row in group]))
        mean_num_steps = float(np.mean([float(row["mean_num_steps"]) for row in group]))
        mean_train_seconds = float(np.mean([float(row["train_seconds"]) for row in group]))

        aggregated_rows.append(
            {
                "case_id": case_id,
                "algo": algo_name,
                "num_train_runs": num_runs,
                "num_successful_trainings": num_successful,
                "percent_successful": percent_successful,
                "mean_success_rate": mean_success_rate,
                "mean_final_distance": mean_final_distance,
                "mean_total_reward": mean_total_reward,
                "mean_num_steps": mean_num_steps,
                "mean_train_seconds": mean_train_seconds,
            }
        )
    return aggregated_rows


def save_aggregate_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    """Save the aggregated percentage table."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    fieldnames = [
        "case_id",
        "algo",
        "num_train_runs",
        "num_successful_trainings",
        "percent_successful",
        "mean_success_rate",
        "mean_final_distance",
        "mean_total_reward",
        "mean_num_steps",
        "mean_train_seconds",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_int_list(text: str) -> List[int]:
    """Convert a comma-separated integer list into Python integers."""
    return [int(item.strip()) for item in text.split(",") if item.strip()]


def parse_str_list(text: str) -> List[str]:
    """Convert a comma-separated string list into lowercase algorithm names."""
    return [item.strip().lower() for item in text.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mass-train several RL policies across case 0 / 1 / 2, evaluate each trained "
            "best policy deterministically, and compute the percentage of training runs that "
            "achieve success_rate = 1.0."
        )
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="0,1,2",
        help="Comma-separated case IDs to process, e.g. 0,1,2",
    )
    parser.add_argument(
        "--algos",
        type=str,
        default="ppo,sac,td3,tqc",
        help="Comma-separated algorithms to process, e.g. ppo,sac,td3,tqc",
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated list of training seeds, e.g. 0,1,2,3,4",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=150_000,
        help="Training budget for every run",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency passed to the training script",
    )
    parser.add_argument(
        "--train-eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes used by the training script callback",
    )
    parser.add_argument(
        "--final-eval-episodes",
        type=int,
        default=10,
        help="Number of deterministic episodes used to judge one trained policy",
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch the per-case training scripts",
    )
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Root directory that contains case0_rl_train.py, case1_rl_train.py, and case2_rl_train.py",
    )
    parser.add_argument(
        "--results-root",
        type=str,
        default="training_stability_results",
        help="Directory where all trained models and CSV summaries will be stored",
    )
    args = parser.parse_args()

    case_ids = parse_int_list(args.cases)
    algo_names = parse_str_list(args.algos)
    seeds = parse_int_list(args.seeds)

    for case_id in case_ids:
        if case_id not in SUPPORTED_CASES:
            raise ValueError(f"Unsupported case ID: {case_id}")
    for algo_name in algo_names:
        if algo_name not in SUPPORTED_ALGOS:
            raise ValueError(f"Unsupported algorithm: {algo_name}")

    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)

    detailed_rows: List[Dict[str, object]] = []

    for case_id in case_ids:
        train_script = project_root / get_train_script_name(case_id)
        if not train_script.exists():
            raise FileNotFoundError(f"Training script not found: {train_script}")

        for algo_name in algo_names:
            for seed in seeds:
                run_dir = results_root / f"case_{case_id}" / algo_name / f"seed_{seed:03d}"
                run_dir.mkdir(parents=True, exist_ok=True)

                train_start = time.perf_counter()
                best_model_path = train_one_run(
                    python_executable=args.python_executable,
                    train_script=train_script,
                    case_id=case_id,
                    algo_name=algo_name,
                    seed=seed,
                    total_timesteps=args.total_timesteps,
                    eval_freq=args.eval_freq,
                    n_eval_episodes=args.train_eval_episodes,
                    output_dir=run_dir,
                )
                train_seconds = time.perf_counter() - train_start

                eval_row = evaluate_policy(
                    case_id=case_id,
                    algo_name=algo_name,
                    model_path=best_model_path,
                    n_eval_episodes=args.final_eval_episodes,
                    deterministic=True,
                )
                eval_row["seed"] = seed
                eval_row["run_dir"] = str(run_dir)
                eval_row["train_seconds"] = train_seconds
                detailed_rows.append(eval_row)

                print(
                    f"[RESULT] case={case_id} algo={algo_name} seed={seed} | "
                    f"success_rate={eval_row['success_rate']:.3f} | "
                    f"successful_training={eval_row['successful_training']}"
                )

    detailed_csv = results_root / "training_stability_detailed.csv"
    aggregate_csv = results_root / "training_stability_aggregate.csv"

    save_detailed_summary_csv(detailed_rows, detailed_csv)
    aggregated_rows = aggregate_results(detailed_rows)
    save_aggregate_csv(aggregated_rows, aggregate_csv)

    print("\n=== Aggregated training stability ===")
    for row in aggregated_rows:
        print(
            f"case={row['case_id']} algo={row['algo']} | "
            f"successful={row['num_successful_trainings']}/{row['num_train_runs']} | "
            f"percent={row['percent_successful']:.2f}%"
        )

    print(f"\nDetailed CSV : {detailed_csv}")
    print(f"Aggregate CSV: {aggregate_csv}")


if __name__ == "__main__":
    main()

