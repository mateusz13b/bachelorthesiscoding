import argparse
from pathlib import Path
from typing import Callable

import gymnasium as gym
import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

try:
    from sb3_contrib import TQC
except ImportError:
    TQC = None


GOAL_POS = np.array([0.4, 0.0, 0.1], dtype=float)
START_JOINTS = np.array([2.7, 0.2, 0.2, -1.0, 0.0, 1.7, 0.7], dtype=float)


class PandaReachCase0Task(Reach):
    """Case 0: fixed end-effector goal with no obstacles."""

    def reset(self) -> None:
        self.goal = GOAL_POS.copy()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


class JointControlPanda(Panda):
    """Panda robot with joint-space control."""

    def __init__(self, sim):
        super().__init__(sim, block_gripper=True, control_type="joints")

    def reset(self) -> None:
        super().reset()
        self.set_joint_angles(START_JOINTS)


def make_env() -> Callable[[], gym.Env]:
    """Create a single training environment for case 0."""

    def _init() -> gym.Env:
        from panda_gym.pybullet import PyBullet

        sim = PyBullet()
        robot = JointControlPanda(sim)
        task = PandaReachCase0Task(
            sim,
            get_ee_position=robot.get_ee_position,
            reward_type="dense",
        )
        env = RobotTaskEnv(robot, task)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=300)
        env = Monitor(env)
        return env

    return _init


def build_model(algo_name: str, env, seed: int | None = None):
    """Build the selected RL model."""
    algo = algo_name.lower()

    if algo == "ppo":
        return PPO(
            "MultiInputPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            verbose=1,
            seed=seed,
        )

    if algo in {"sac", "td3", "tqc"}:
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=0.10 * np.ones(n_actions),
        )

        common_kwargs = dict(
            policy="MultiInputPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=300_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            verbose=1,
            seed=seed,
        )

        if algo == "sac":
            return SAC(**common_kwargs)

        if algo == "td3":
            return TD3(
                **common_kwargs,
                action_noise=action_noise,
                policy_delay=2,
            )

        if algo == "tqc":
            if TQC is None:
                raise ImportError(
                    "The --algo tqc option requires sb3-contrib: python -m pip install sb3-contrib"
                )
            return TQC(
                **common_kwargs,
                top_quantiles_to_drop_per_net=2,
            )

    raise ValueError("Supported algorithms: ppo, sac, td3, tqc")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an RL policy for Panda case 0 with joint-space control."
    )
    parser.add_argument(
        "--algo",
        required=True,
        choices=["ppo", "sac", "td3", "tqc"],
        help="Training algorithm",
    )
    parser.add_argument(
        "--case-id",
        type=int,
        default=0,
        help="Case ID used in saved file names. Default: 0",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=150_000, #timesteps for learning
        help="Total number of training timesteps",
    )
    parser.add_argument(
        "--eval-freq",
        type=int,
        default=10_000,
        help="Evaluation frequency in timesteps",
    )
    parser.add_argument(
        "--n-eval-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Training seed",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where models will be saved",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv([make_env()])
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv([make_env()])
    eval_env = VecMonitor(eval_env)

    model = build_model(args.algo, train_env, seed=args.seed)

    best_model_path = output_dir / f"case_{args.case_id}_{args.algo}_best_policy"
    final_model_path = output_dir / f"case_{args.case_id}_{args.algo}_final_policy"
    eval_log_path = output_dir / f"case_{args.case_id}_{args.algo}_eval"

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        log_path=str(eval_log_path),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.n_eval_episodes,
        deterministic=True,
        render=False,
        verbose=1,
    )

    print(f"=== Training {args.algo.upper()} for case {args.case_id} ===")
    print(f"Goal position: {GOAL_POS}")
    print(f"Start joints: {START_JOINTS}")

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(str(final_model_path))

    # EvalCallback saves the best model as best_model.zip.
    # Rename it to match the project naming convention if it exists.
    default_best_zip = output_dir / "best_model.zip"
    target_best_zip = Path(str(best_model_path) + ".zip")
    if default_best_zip.exists():
        default_best_zip.replace(target_best_zip)
        print(f"Best policy saved to: {target_best_zip}")
    else:
        print("EvalCallback did not save a best model; only the final policy is available.")

    print(f"Final policy saved to: {final_model_path}.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()

