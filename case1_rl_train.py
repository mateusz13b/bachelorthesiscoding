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
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

try:
    from sb3_contrib import TQC
except ImportError:  # pragma: no cover
    TQC = None


CASE_ID = 1
GOAL_POSITION = np.array([0.4, 0.0, 0.1], dtype=np.float32)
START_JOINTS = np.array([2.7, 0.2, 0.2, -1.0, 0.0, 1.7, 0.7], dtype=np.float32)
DEFAULT_DISTANCE_THRESHOLD = 0.01
DEFAULT_MAX_EPISODE_STEPS = 300
DEFAULT_REWARD_TYPE = "dense"

WALL_X = 0.20
WALL_WIDTH = 0.01
WALL_HALF_Y = 0.20
WALL_HEIGHT = 0.20


class PandaReachWallCase1Task(Reach):
    """Fixed EE goal with a thin vertical wall at x=0.20."""

    def __init__(self, sim, get_ee_position, reward_type: str = DEFAULT_REWARD_TYPE):
        super().__init__(
            sim,
            get_ee_position=get_ee_position,
            reward_type=reward_type,
            distance_threshold=DEFAULT_DISTANCE_THRESHOLD,
        )
        self.sim.create_box(
            body_name="wall",
            half_extents=[WALL_WIDTH / 2.0, WALL_HALF_Y, WALL_HEIGHT],
            mass=0.0,
            position=[WALL_X, 0.0, WALL_HEIGHT / 2.0],
            rgba_color=[0.8, 0.1, 0.1, 1.0],
        )

    def reset(self):
        self.goal = GOAL_POSITION.copy()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


class JointControlPanda(Panda):
    """Franka Panda with joint-space actions."""

    def __init__(self, sim):
        super().__init__(sim, block_gripper=True, control_type="joints")

    def reset(self):
        super().reset()
        self.set_joint_angles(START_JOINTS)


def make_env(
    max_episode_steps: int = DEFAULT_MAX_EPISODE_STEPS,
    reward_type: str = DEFAULT_REWARD_TYPE,
) -> Callable[[], gym.Env]:
    def _init() -> gym.Env:
        from panda_gym.pybullet import PyBullet

        sim = PyBullet()
        robot = JointControlPanda(sim)
        task = PandaReachWallCase1Task(
            sim,
            get_ee_position=robot.get_ee_position,
            reward_type=reward_type,
        )
        env = RobotTaskEnv(robot, task)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        return env

    return _init


def build_algorithm(
    algo_name: str,
    env,
    learning_rate: float,
    buffer_size: int,
    learning_starts: int,
    batch_size: int,
    gamma: float,
    tau: float,
    policy_delay: int,
    action_noise_std: float,
    seed: int,
    verbose: int,
):
    algo_key = algo_name.lower()

    if algo_key == "ppo":
        return PPO(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            n_steps=2048,
            batch_size=64,
            gamma=gamma,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            seed=seed,
            verbose=verbose,
        )

    if algo_key == "sac":
        return SAC(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            seed=seed,
            verbose=verbose,
        )

    if algo_key == "td3":
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions),
        )
        return TD3(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            action_noise=action_noise,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            policy_delay=policy_delay,
            seed=seed,
            verbose=verbose,
        )

    if algo_key == "tqc":
        if TQC is None:
            raise ImportError(
                "TQC requires sb3-contrib. Install it with: pip install sb3-contrib"
            )
        return TQC(
            "MultiInputPolicy",
            env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=1,
            gradient_steps=1,
            top_quantiles_to_drop_per_net=2,
            seed=seed,
            verbose=verbose,
        )

    raise ValueError(f"Unsupported algorithm: {algo_name}")


def maybe_standardize_best_model(output_dir: Path, algo_name: str) -> Path | None:
    raw_best_model = output_dir / "best_model.zip"
    if not raw_best_model.exists():
        return None

    standardized_best_model = output_dir / f"case_{CASE_ID}_{algo_name.lower()}_best_policy.zip"
    if standardized_best_model.exists():
        standardized_best_model.unlink()
    raw_best_model.replace(standardized_best_model)
    return standardized_best_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train PPO/SAC/TD3/TQC for Panda case 1 (wall) with joint-space control."
    )
    parser.add_argument("--algo", required=True, choices=["ppo", "sac", "td3", "tqc"], help="RL algorithm")
    parser.add_argument("--total-timesteps", type=int, default=100_000, help="Training timesteps")
    parser.add_argument("--save-dir", type=str, default="models", help="Directory for saved policies")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--eval-freq", type=int, default=5_000, help="Evaluation frequency in env steps")
    parser.add_argument("--n-eval-episodes", type=int, default=5, help="Episodes per evaluation")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--buffer-size", type=int, default=300_000, help="Replay buffer size for off-policy methods")
    parser.add_argument("--learning-starts", type=int, default=5_000, help="Warmup steps for off-policy methods")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for off-policy methods")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--tau", type=float, default=0.005, help="Soft-update coefficient for off-policy methods")
    parser.add_argument("--policy-delay", type=int, default=2, help="TD3 actor update delay")
    parser.add_argument("--action-noise-std", type=float, default=0.15, help="TD3 Gaussian action-noise std")
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
    parser.add_argument("--verbose", type=int, default=1, help="SB3 verbosity")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    algo_name = args.algo.lower()
    output_dir = Path(args.save_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_env = DummyVecEnv(
        [
            make_env(
                max_episode_steps=args.max_episode_steps,
                reward_type=args.reward_type,
            )
        ]
    )
    train_env = VecMonitor(train_env)

    eval_env = DummyVecEnv(
        [
            make_env(
                max_episode_steps=args.max_episode_steps,
                reward_type=args.reward_type,
            )
        ]
    )
    eval_env = VecMonitor(eval_env)

    model = build_algorithm(
        algo_name=algo_name,
        env=train_env,
        learning_rate=args.learning_rate,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        batch_size=args.batch_size,
        gamma=args.gamma,
        tau=args.tau,
        policy_delay=args.policy_delay,
        action_noise_std=args.action_noise_std,
        seed=args.seed,
        verbose=args.verbose,
    )

    final_model_path = output_dir / f"case_{CASE_ID}_{algo_name}_final_policy"
    raw_best_model = output_dir / "best_model.zip"
    if raw_best_model.exists():
        raw_best_model.unlink()

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir),
        n_eval_episodes=args.n_eval_episodes,
        eval_freq=args.eval_freq,
        deterministic=True,
        render=False,
        verbose=args.verbose,
    )

    print(f"=== Training {algo_name.upper()} — Case {CASE_ID} wall reach, joint control ===")
    print(f"Goal position: {GOAL_POSITION.tolist()}")
    print(f"Wall x-position: {WALL_X}")
    print(f"Saving final model to: {final_model_path}.zip")

    model.learn(total_timesteps=args.total_timesteps, callback=eval_callback)
    model.save(str(final_model_path))

    best_model_path = maybe_standardize_best_model(output_dir, algo_name)

    train_env.close()
    eval_env.close()

    print("\nTraining finished.")
    print(f"Final policy: {final_model_path}.zip")
    if best_model_path is not None:
        print(f"Best policy:  {best_model_path}")
    else:
        print("Best policy was not created. Check eval settings and training run.")


if __name__ == "__main__":
    main()
