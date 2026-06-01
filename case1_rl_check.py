import argparse
import csv
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import gymnasium as gym
import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from stable_baselines3 import PPO, SAC, TD3

try:
    from sb3_contrib import TQC
except ImportError:  # pragma: no cover
    TQC = None

from case1_rl_train import (
    CASE_ID,
    DEFAULT_MAX_EPISODE_STEPS,
    DEFAULT_REWARD_TYPE,
    JointControlPanda,
    PandaReachWallCase1Task,
)


def get_model_class(algo_name: str):
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
                "TQC requires sb3-contrib. Install it with: pip install sb3-contrib"
            )
        return TQC
    raise ValueError(f"Unsupported algorithm: {algo_name}")


def resolve_model_path(algo_name: str, model_path: str | None, save_dir: str) -> Path:
    if model_path is not None:
        resolved = Path(model_path)
    else:
        resolved = Path(save_dir) / f"case_{CASE_ID}_{algo_name.lower()}_best_policy.zip"

    if not resolved.exists():
        raise FileNotFoundError(f"Model file not found: {resolved}")
    return resolved


def resolve_log_paths(
    algo_name: str,
    episode_tag: str,
    output_dir: str,
    case_id: int = CASE_ID,
) -> Tuple[Path, Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    prefix = f"case_{case_id}_{algo_name.lower()}_{episode_tag}"
    return output_path / f"{prefix}_ee.csv", output_path / f"{prefix}_joints.csv"


def make_single_env(
    render: bool = True,
    reward_type: str = DEFAULT_REWARD_TYPE,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
):
    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render_mode="human") if render else PyBullet()
    robot = JointControlPanda(sim)
    task = PandaReachWallCase1Task(
        sim,
        get_ee_position=robot.get_ee_position,
        reward_type=reward_type,
    )
    env = RobotTaskEnv(robot, task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    return env, robot


def get_arm_joint_angles(robot: JointControlPanda) -> np.ndarray:
    return np.array([robot.get_joint_angle(joint=i) for i in range(7)], dtype=np.float32)


def collect_initial_state(robot: JointControlPanda) -> Dict[str, np.ndarray]:
    return {
        "joint": get_arm_joint_angles(robot).copy(),
        "ee_pos": robot.get_ee_position().copy(),
        "ee_vel": robot.get_ee_velocity().copy(),
    }


def save_ee_log(ee_history: np.ndarray, ee_velocity_history: np.ndarray, log_path: Path, log_dt: float) -> None:
    with log_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["timestep", "time_sec", "posx", "posy", "posz", "velx", "vely", "velz"])
        for timestep, (pos, vel) in enumerate(zip(ee_history, ee_velocity_history)):
            writer.writerow([
                timestep,
                round(timestep * log_dt, 6),
                round(float(pos[0]), 6),
                round(float(pos[1]), 6),
                round(float(pos[2]), 6),
                round(float(vel[0]), 6),
                round(float(vel[1]), 6),
                round(float(vel[2]), 6),
            ])


def save_joint_log(joint_history: np.ndarray, log_path: Path, log_dt: float) -> None:
    header = ["timestep", "time_sec"] + [f"j{i}" for i in range(1, 8)]
    with log_path.open("w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(header)
        for timestep, joints in enumerate(joint_history):
            writer.writerow([
                timestep,
                round(timestep * log_dt, 6),
                *[round(float(angle), 6) for angle in joints],
            ])


def save_run_logs(
    algo_name: str,
    episode_tag: str,
    output_dir: str,
    joint_history: np.ndarray,
    ee_history: np.ndarray,
    ee_velocity_history: np.ndarray,
    log_dt: float,
    case_id: int = CASE_ID,
) -> Tuple[Path, Path]:
    ee_path, joint_path = resolve_log_paths(
        algo_name=algo_name,
        episode_tag=episode_tag,
        output_dir=output_dir,
        case_id=case_id,
    )
    save_ee_log(ee_history=ee_history, ee_velocity_history=ee_velocity_history, log_path=ee_path, log_dt=log_dt)
    save_joint_log(joint_history=joint_history, log_path=joint_path, log_dt=log_dt)
    return ee_path, joint_path


def run_policy_once(
    model,
    render: bool = True,
    deterministic: bool = True,
    reward_type: str = DEFAULT_REWARD_TYPE,
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    sleep_dt: float = 1 / 60,
    print_every: int = 50,
) -> Dict[str, Any]:
    env, robot = make_single_env(
        render=render,
        reward_type=reward_type,
        max_episode_steps=max_episode_steps,
    )
    obs, info = env.reset()

    joint_history = []
    ee_history = []
    ee_velocity_history = []
    rewards = []

    initial_state = collect_initial_state(robot)
    joint_history.append(initial_state["joint"])
    ee_history.append(initial_state["ee_pos"])
    ee_velocity_history.append(initial_state["ee_vel"])

    terminated = False
    truncated = False
    step_idx = 0

    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        step_idx += 1

        joint_history.append(get_arm_joint_angles(robot).copy())
        ee_history.append(robot.get_ee_position().copy())
        ee_velocity_history.append(robot.get_ee_velocity().copy())

        if print_every > 0 and step_idx % print_every == 0:
            achieved = obs["achieved_goal"]
            desired = obs["desired_goal"]
            distance = float(np.linalg.norm(achieved - desired))
            print(
                f"step {step_idx:4d} | distance: {distance:.4f} m "
                f"| success: {bool(info.get('is_success', False))}"
            )

        if render:
            time.sleep(sleep_dt)

    env.close()

    return {
        "joint_history": np.asarray(joint_history, dtype=np.float32),
        "ee_history": np.asarray(ee_history, dtype=np.float32),
        "ee_velocity_history": np.asarray(ee_velocity_history, dtype=np.float32),
        "total_reward": float(np.sum(rewards)),
        "num_steps": int(step_idx),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "success": bool(info.get("is_success", False)),
        "final_distance": float(np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])),
    }


def load_policy(algo_name: str, model_path: str | Path):
    model_class = get_model_class(algo_name)
    return model_class.load(str(model_path))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load and run the best PPO/SAC/TD3/TQC policy for case 1 with human rendering and CSV logging."
    )
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "td3", "tqc"], help="RL algorithm")
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to a saved best policy (.zip). If omitted, uses models/case_1_<algo>_best_policy.zip",
    )
    parser.add_argument("--save-dir", type=str, default="models", help="Directory with saved best policies")
    parser.add_argument(
        "--episode-tag",
        type=str,
        default="episode",
        help="Tag used in output CSV filenames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where EE and joint CSV logs will be saved",
    )
    parser.add_argument(
        "--log-dt",
        type=float,
        default=0.02,
        help="Time step used for time_sec in saved CSV logs",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample stochastic actions. Use this when a comparative script runs the policy multiple times.",
    )
    parser.add_argument(
        "--reward-type",
        type=str,
        default=DEFAULT_REWARD_TYPE,
        choices=["dense", "sparse"],
        help="Task reward type",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=DEFAULT_MAX_EPISODE_STEPS,
        help="Episode horizon",
    )
    parser.add_argument(
        "--sleep-dt",
        type=float,
        default=1 / 60,
        help="Delay between rendered steps in seconds",
    )
    parser.add_argument(
        "--print-every",
        type=int,
        default=50,
        help="Print progress every N steps; use 0 to disable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_path = resolve_model_path(args.algo, args.model, args.save_dir)
    model = load_policy(args.algo, model_path)

    print(f"Running {args.algo.upper()} policy from: {model_path}")
    deterministic = not args.stochastic
    print(f"Case: {CASE_ID} | deterministic={deterministic}")

    result = run_policy_once(
        model=model,
        render=True,
        deterministic=deterministic,
        reward_type=args.reward_type,
        max_episode_steps=args.max_episode_steps,
        sleep_dt=args.sleep_dt,
        print_every=args.print_every,
    )

    ee_path, joint_path = save_run_logs(
        algo_name=args.algo,
        episode_tag=args.episode_tag,
        output_dir=args.output_dir,
        joint_history=result["joint_history"],
        ee_history=result["ee_history"],
        ee_velocity_history=result["ee_velocity_history"],
        log_dt=args.log_dt,
    )

    print("\nRun finished.")
    print(f"Success:        {result['success']}")
    print(f"Steps:          {result['num_steps']}")
    print(f"Total reward:   {result['total_reward']:.4f}")
    print(f"Final distance: {result['final_distance']:.6f} m")
    print(f"EE log saved:   {ee_path}")
    print(f"Joint log saved:{joint_path}")


if __name__ == "__main__":
    main()

