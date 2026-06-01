import argparse
import csv
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach

#python trajopt_case2_standardized.py --render

CASE_ID = 2
DEFAULT_ALGO_NAME = "trajopt"
DEFAULT_EPISODE_TAG = "episode"
DEFAULT_SOURCE_DT = 1.0 / 240.0
DEFAULT_LOG_DT = 0.02
START_JOINTS = np.array([2.7, 0.2, 0.2, -1.0, 0.0, 1.7, 0.7], dtype=float)
GOAL_POS = np.array([0.4, 0.0, 0.1], dtype=float)


class PandaReachCase2Task(Reach):
    """Case 2: fixed end-effector goal with a high wall obstacle."""

    def __init__(self, sim, get_ee_position, reward_type: str = "dense"):
        super().__init__(
            sim,
            get_ee_position=get_ee_position,
            reward_type=reward_type,
            distance_threshold=0.01,
        )

        self.wall_x = 0.35
        self.wall_width = 0.02
        self.gap_height = 0.30
        self.wall_height = 0.5
        self.wall_width_y = 1.5

        z_center = self.gap_height + self.wall_height / 2.0

        self.sim.create_box(
            body_name="high_wall",
            half_extents=[
                self.wall_width / 2.0,
                self.wall_width_y / 2.0,
                self.wall_height / 2.0,
            ],
            mass=0,
            position=[self.wall_x, 0.0, z_center],
            rgba_color=[0.8, 0.1, 0.1, 0.8],
        )

    def reset(self):
        self.goal = GOAL_POS.copy()
        self.sim.set_base_pose("target", self.goal, np.array([0.0, 0.0, 0.0, 1.0]))


class CustomPandaRobot(Panda):
    """Panda robot with joint-space control."""

    def __init__(self, sim):
        super().__init__(sim, block_gripper=True, control_type="joints")

    def reset(self):
        super().reset()
        self.set_joint_angles(START_JOINTS)


def build_env(render: bool) -> Tuple[RobotTaskEnv, Panda, object]:
    """Build a case 2 environment and return environment, robot, and simulator."""
    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render_mode="human") if render else PyBullet()
    robot = CustomPandaRobot(sim)
    task = PandaReachCase2Task(sim, robot.get_ee_position)
    env = RobotTaskEnv(robot, task)
    return env, robot, sim


def get_arm_joint_indices(robot) -> list:
    """Return the first seven Panda arm joint indices."""
    return list(robot.joint_indices[:7])


def get_current_arm_joint_angles(robot) -> np.ndarray:
    """Return the seven Panda arm joint angles in radians."""
    arm_joint_indices = get_arm_joint_indices(robot)
    return np.array([robot.get_joint_angle(int(j)) for j in arm_joint_indices], dtype=float)


def safe_set_joint_angles(robot, q: np.ndarray) -> None:
    """Set joint angles while handling APIs that expect a full joint vector."""
    q = np.asarray(q, dtype=float)
    try:
        robot.set_joint_angles(q)
    except Exception:
        full_q = np.zeros(len(robot.joint_indices), dtype=float)
        full_q[: len(q)] = q
        robot.set_joint_angles(full_q)


def safe_control_joints(robot, q: np.ndarray) -> None:
    """Send a joint-space command while handling APIs that expect a full joint vector."""
    q = np.asarray(q, dtype=float)
    try:
        robot.control_joints(q)
    except Exception:
        full_q = np.zeros(len(robot.joint_indices), dtype=float)
        full_q[: len(q)] = q
        robot.control_joints(full_q)


def smooth_trajectory(traj: np.ndarray, passes: int = 4, lam: float = 0.20) -> np.ndarray:
    """Apply simple post-smoothing while keeping endpoints fixed."""
    out = traj.copy()
    for _ in range(passes):
        new_out = out.copy()
        for i in range(1, len(out) - 1):
            new_out[i] = (1.0 - lam) * out[i] + 0.5 * lam * (out[i - 1] + out[i + 1])
        new_out[0] = out[0]
        new_out[-1] = out[-1]
        out = new_out
    return out


class GoalIKSolver:
    """Numerical solver for a final joint configuration that reaches the Cartesian goal."""

    def __init__(
        self,
        robot,
        start_q: np.ndarray,
        goal_pos: np.ndarray,
        regularization_weight: float = 1e-3,
    ):
        self.robot = robot
        self.start_q = np.array(start_q[:7], dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)
        self.n_dof = 7
        self.regularization_weight = regularization_weight
        self.q_min = np.array([-2.9, -1.8, -2.9, -3.0, -2.9, -0.1, -2.9], dtype=float)
        self.q_max = np.array([2.9, 1.8, 2.9, -0.05, 2.9, 3.7, 2.9], dtype=float)

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_min, self.q_max)

    def get_ee_pos_for_q(self, q: np.ndarray) -> np.ndarray:
        safe_set_joint_angles(self.robot, q)
        return self.robot.get_ee_position().copy()

    def goal_cost(self, q: np.ndarray) -> float:
        ee = self.get_ee_pos_for_q(q)
        pos_err = ee - self.goal_pos
        reg = self.regularization_weight * np.sum((q - self.start_q) ** 2)
        return 0.5 * float(np.dot(pos_err, pos_err)) + reg

    def numerical_grad_goal(self, q: np.ndarray, eps: float = 2e-3) -> np.ndarray:
        grad = np.zeros(self.n_dof, dtype=float)
        for j in range(self.n_dof):
            q_p = q.copy()
            q_m = q.copy()
            q_p[j] += eps
            q_m[j] -= eps
            q_p = self.clip_q(q_p)
            q_m = self.clip_q(q_m)
            c_p = self.goal_cost(q_p)
            c_m = self.goal_cost(q_m)
            grad[j] = (c_p - c_m) / (2.0 * eps)
        return grad

    def solve(self, max_iters: int = 260, alpha: float = 0.12) -> np.ndarray:
        seeds = [
            self.start_q.copy(),
            self.start_q + np.array([0.55, 0.25, 0.00, 0.15, 0.00, 0.00, 0.00]),
            self.start_q + np.array([0.35, -0.20, 0.20, 0.20, 0.00, 0.20, 0.00]),
            self.start_q + np.array([-0.35, 0.25, -0.10, -0.10, 0.00, 0.00, 0.00]),
            np.array([1.6, 0.8, -0.2, -1.9, -0.2, 2.4, 0.8], dtype=float),
        ]

        best_q = None
        best_dist = np.inf

        for seed in seeds:
            q = self.clip_q(seed)
            for _ in range(max_iters):
                ee = self.get_ee_pos_for_q(q)
                err = ee - self.goal_pos
                if np.linalg.norm(err) < 0.008:
                    break
                grad = self.numerical_grad_goal(q)
                q = self.clip_q(q - alpha * grad)

            final_dist = np.linalg.norm(self.get_ee_pos_for_q(q) - self.goal_pos)
            if final_dist < best_dist:
                best_dist = final_dist
                best_q = q.copy()

        print(f"[IK] best final EE error: {best_dist:.6f}")
        return best_q


class JointSpaceTrajOptPlanner:
    """A simple deterministic TrajOpt-style joint-space optimizer for case 2."""

    def __init__(
        self,
        robot,
        start_q: np.ndarray,
        goal_pos: np.ndarray,
        wall_x: float,
        wall_width: float = 0.02,
        wall_width_y: float = 1.5,
        wall_height: float = 0.5,
        gap_height: float = 0.30,
        num_timesteps: int = 32,
        num_iterations: int = 120,
        trust_region: float = 0.16,
        smoothness_weight: float = 1.0,
        obstacle_weight: float = 24.0,
        goal_weight: float = 8.0,
        regularization_weight: float = 1e-3,
        safety_margin: float = 0.06,
        initial_dip: float = 0.35,
    ):
        self.robot = robot
        self.arm_joint_indices = get_arm_joint_indices(robot)
        self.start_q = np.array(start_q[:7], dtype=float)
        self.goal_pos = np.array(goal_pos, dtype=float)

        self.n_dof = 7
        self.T = num_timesteps
        self.iterations = num_iterations
        self.trust_region = trust_region
        self.smoothness_weight = smoothness_weight
        self.obstacle_weight = obstacle_weight
        self.goal_weight = goal_weight
        self.regularization_weight = regularization_weight
        self.safety_margin = safety_margin
        self.initial_dip = initial_dip

        self.q_min = np.array([-2.9, -1.8, -2.9, -3.0, -2.9, -0.1, -2.9], dtype=float)
        self.q_max = np.array([2.9, 1.8, 2.9, -0.05, 2.9, 3.7, 2.9], dtype=float)

        wall_center_z = gap_height + wall_height / 2.0
        self.wall_center = np.array([wall_x, 0.0, wall_center_z], dtype=float)
        self.wall_half_extents = np.array([wall_width / 2.0, wall_width_y / 2.0, wall_height / 2.0], dtype=float)

    def clip_q(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.q_min, self.q_max)

    def set_q(self, q: np.ndarray) -> None:
        safe_set_joint_angles(self.robot, q)

    def get_ee_pos_for_q(self, q: np.ndarray) -> np.ndarray:
        self.set_q(q)
        return self.robot.get_ee_position().copy()

    def get_robot_points_for_q(self, q: np.ndarray) -> np.ndarray:
        self.set_q(q)
        pts = []
        for j in self.arm_joint_indices:
            try:
                pts.append(self.robot.get_link_position(int(j)).copy())
            except Exception:
                pass
        pts.append(self.robot.get_ee_position().copy())
        return np.array(pts, dtype=float)

    def signed_distance_to_wall_box(self, p: np.ndarray) -> float:
        rel = p - self.wall_center
        q = np.abs(rel) - self.wall_half_extents
        outside = np.maximum(q, 0.0)
        outside_dist = np.linalg.norm(outside)
        if outside_dist > 1e-12:
            return float(outside_dist)
        margins = self.wall_half_extents - np.abs(rel)
        return -float(np.min(margins))

    def point_obstacle_cost(self, p: np.ndarray) -> float:
        d = self.signed_distance_to_wall_box(p)
        eps = self.safety_margin
        if d >= eps:
            return 0.0
        if d >= 0.0:
            return 0.5 * ((eps - d) ** 2) / eps
        return -d + 0.5 * eps

    def configuration_obstacle_cost(self, q: np.ndarray) -> float:
        pts = self.get_robot_points_for_q(q)
        return float(sum(self.point_obstacle_cost(p) for p in pts))

    def waypoint_goal_cost(self, q: np.ndarray) -> float:
        ee = self.get_ee_pos_for_q(q)
        pos_err = ee - self.goal_pos
        reg = self.regularization_weight * np.sum((q - self.start_q) ** 2)
        return 0.5 * float(np.dot(pos_err, pos_err)) + reg

    def smoothness_cost(self, traj: np.ndarray) -> float:
        total = 0.0
        for i in range(1, self.T - 1):
            acc = traj[i - 1] - 2.0 * traj[i] + traj[i + 1]
            total += 0.5 * float(np.dot(acc, acc))
        return total

    def obstacle_cost(self, traj: np.ndarray) -> float:
        return float(sum(self.configuration_obstacle_cost(traj[i]) for i in range(1, self.T - 1)))

    def terminal_goal_cost(self, traj: np.ndarray) -> float:
        return self.waypoint_goal_cost(traj[-1])

    def total_cost(self, traj: np.ndarray) -> float:
        return (
            self.smoothness_weight * self.smoothness_cost(traj)
            + self.obstacle_weight * self.obstacle_cost(traj)
            + self.goal_weight * self.terminal_goal_cost(traj)
        )

    def numerical_waypoint_gradient(self, traj: np.ndarray, waypoint_idx: int, eps: float = 2e-3) -> np.ndarray:
        grad = np.zeros(self.n_dof, dtype=float)
        base_q = traj[waypoint_idx].copy()
        for j in range(self.n_dof):
            traj_p = traj.copy()
            traj_m = traj.copy()
            traj_p[waypoint_idx] = base_q.copy()
            traj_m[waypoint_idx] = base_q.copy()
            traj_p[waypoint_idx, j] += eps
            traj_m[waypoint_idx, j] -= eps
            traj_p[waypoint_idx] = self.clip_q(traj_p[waypoint_idx])
            traj_m[waypoint_idx] = self.clip_q(traj_m[waypoint_idx])
            c_p = self.total_cost(traj_p)
            c_m = self.total_cost(traj_m)
            grad[j] = (c_p - c_m) / (2.0 * eps)
        return grad

    def initialize_trajectory(self, q_goal: np.ndarray) -> np.ndarray:
        traj = np.linspace(self.start_q, q_goal, self.T)
        s = np.linspace(0.0, 1.0, self.T)
        traj[:, 1] += 0.10 * np.sin(np.pi * s)
        traj[:, 3] -= self.initial_dip * np.sin(np.pi * s)
        traj[0] = self.start_q
        traj[-1] = q_goal
        return self.clip_q(traj)

    def plan(self) -> np.ndarray:
        ik_solver = GoalIKSolver(
            robot=self.robot,
            start_q=self.start_q,
            goal_pos=self.goal_pos,
            regularization_weight=self.regularization_weight,
        )
        q_goal = ik_solver.solve()
        trajectory = self.initialize_trajectory(q_goal)
        best_traj = trajectory.copy()
        best_cost = self.total_cost(best_traj)

        print(f"[TrajOpt] initial total cost: {best_cost:.6f}")

        for it in range(self.iterations):
            updated = trajectory.copy()
            for waypoint_idx in range(1, self.T - 1):
                grad = self.numerical_waypoint_gradient(trajectory, waypoint_idx)
                step = -self.trust_region * np.tanh(grad)
                updated[waypoint_idx] = self.clip_q(trajectory[waypoint_idx] + step)

            updated[0] = self.start_q
            updated[-1] = q_goal
            updated = smooth_trajectory(updated, passes=2, lam=0.18)
            updated[0] = self.start_q
            updated[-1] = q_goal

            new_cost = self.total_cost(updated)
            old_cost = self.total_cost(trajectory)
            if new_cost < old_cost:
                trajectory = updated
                self.trust_region = min(self.trust_region * 1.03, 0.22)
            else:
                self.trust_region = max(self.trust_region * 0.55, 0.015)

            total = self.total_cost(trajectory)
            if total < best_cost:
                best_cost = total
                best_traj = trajectory.copy()

            if it % 10 == 0 or it == self.iterations - 1:
                s_cost = self.smoothness_cost(trajectory)
                o_cost = self.obstacle_cost(trajectory)
                g_cost = self.terminal_goal_cost(trajectory)
                print(
                    f"[TrajOpt] iter={it:03d} | smooth={s_cost:.6f} | "
                    f"obstacle={o_cost:.6f} | terminal={g_cost:.6f} | total={total:.6f}"
                )

        best_traj = smooth_trajectory(best_traj, passes=6, lam=0.18)
        best_traj[0] = self.start_q
        best_traj[-1] = q_goal
        return best_traj


def build_output_paths(case_id: int, algo_name: str, episode_tag: str, output_dir: Path) -> Tuple[Path, Path]:
    """Build output paths for EE and joint CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"case_{case_id}_{algo_name}_{episode_tag}"
    ee_path = output_dir / f"{prefix}_ee.csv"
    joint_path = output_dir / f"{prefix}_joints.csv"
    return ee_path, joint_path


def collect_state(robot) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect joint angles, EE position, and EE velocity."""
    q = get_current_arm_joint_angles(robot).copy()
    ee_pos = robot.get_ee_position().copy()
    ee_vel = robot.get_ee_velocity().copy()
    return q, ee_pos, ee_vel


def resample_logs(raw_times: np.ndarray, ee_data: np.ndarray, joint_data: np.ndarray, log_dt: float):
    """Resample raw execution logs to a uniform time grid."""
    if len(raw_times) == 0:
        return np.array([]), np.empty((0, 6)), np.empty((0, 7))

    final_time = raw_times[-1]
    target_times = np.arange(0.0, final_time + 1e-12, log_dt)
    if target_times.size == 0 or not np.isclose(target_times[-1], final_time):
        target_times = np.append(target_times, final_time)

    ee_resampled = np.column_stack([
        np.interp(target_times, raw_times, ee_data[:, col]) for col in range(ee_data.shape[1])
    ])
    joint_resampled = np.column_stack([
        np.interp(target_times, raw_times, joint_data[:, col]) for col in range(joint_data.shape[1])
    ])
    return target_times, ee_resampled, joint_resampled


def save_ee_csv(filename: Path, times: np.ndarray, ee_data: np.ndarray) -> None:
    """Save end-effector data in the agreed project format."""
    with filename.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestep", "time_sec", "posx", "posy", "posz", "velx", "vely", "velz"])
        for timestep, (time_sec, row) in enumerate(zip(times, ee_data)):
            writer.writerow([timestep, round(float(time_sec), 6), *[round(float(v), 6) for v in row]])


def save_joint_csv(filename: Path, times: np.ndarray, joint_data: np.ndarray) -> None:
    """Save joint angles in the agreed project format."""
    with filename.open(mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestep", "time_sec", "j1", "j2", "j3", "j4", "j5", "j6", "j7"])
        for timestep, (time_sec, row) in enumerate(zip(times, joint_data)):
            writer.writerow([timestep, round(float(time_sec), 6), *[round(float(v), 6) for v in row]])


def execute_planned_trajectory(
    robot,
    sim,
    planned_q_path: np.ndarray,
    source_dt: float,
    log_dt: float,
    render: bool,
    interpolation_substeps: int = 12,
    controller_inner_steps: int = 3,
    tracking_gain: float = 0.28,
):
    """Execute the planned joint trajectory and return uniform EE and joint logs."""
    raw_times = []
    ee_rows = []
    joint_rows = []

    q, ee_pos, ee_vel = collect_state(robot)
    raw_times.append(0.0)
    ee_rows.append(np.concatenate([ee_pos, ee_vel]))
    joint_rows.append(q)

    sim_step_index = 0

    for i in range(len(planned_q_path) - 1):
        q_from = planned_q_path[i]
        q_to = planned_q_path[i + 1]

        for alpha in np.linspace(0.0, 1.0, interpolation_substeps, endpoint=False)[1:]:
            q_ref = (1.0 - alpha) * q_from + alpha * q_to
            for _ in range(controller_inner_steps):
                current_q = get_current_arm_joint_angles(robot)
                cmd_q = current_q + tracking_gain * (q_ref - current_q)
                safe_control_joints(robot, cmd_q)
                sim.step()
                sim_step_index += 1
                if render:
                    time.sleep(source_dt)
                q, ee_pos, ee_vel = collect_state(robot)
                raw_times.append(sim_step_index * source_dt)
                ee_rows.append(np.concatenate([ee_pos, ee_vel]))
                joint_rows.append(q)

        for _ in range(10):
            current_q = get_current_arm_joint_angles(robot)
            if np.linalg.norm(q_to - current_q) < 0.02:
                break
            cmd_q = current_q + tracking_gain * (q_to - current_q)
            safe_control_joints(robot, cmd_q)
            sim.step()
            sim_step_index += 1
            if render:
                time.sleep(source_dt)
            q, ee_pos, ee_vel = collect_state(robot)
            raw_times.append(sim_step_index * source_dt)
            ee_rows.append(np.concatenate([ee_pos, ee_vel]))
            joint_rows.append(q)

    raw_times = np.asarray(raw_times, dtype=float)
    ee_rows = np.asarray(ee_rows, dtype=float)
    joint_rows = np.asarray(joint_rows, dtype=float)
    return resample_logs(raw_times, ee_rows, joint_rows, log_dt)


def plan_trajopt_path(render_planning: bool = False, seed: int | None = None) -> np.ndarray:
    """Plan a TrajOpt-style joint-space trajectory for case 2."""
    if seed is not None:
        np.random.seed(seed)

    env, robot, _ = build_env(render=render_planning)
    obs, _ = env.reset()
    start_q = get_current_arm_joint_angles(robot)
    goal_pos = obs["desired_goal"]

    planner = JointSpaceTrajOptPlanner(
        robot=robot,
        start_q=start_q,
        goal_pos=goal_pos,
        wall_x=0.35,
        wall_width=0.02,
        wall_width_y=1.5,
        wall_height=0.5,
        gap_height=0.30,
        num_timesteps=32,
        num_iterations=120,
        trust_region=0.16,
        smoothness_weight=1.0,
        obstacle_weight=24.0,
        goal_weight=8.0,
        regularization_weight=1e-3,
        safety_margin=0.06,
        initial_dip=0.35,
    )
    planned_q_path = planner.plan()
    env.close()
    return planned_q_path


def run_trajopt_case2(
    render: bool,
    case_id: int,
    algo_name: str,
    episode_tag: str,
    output_dir: Path,
    source_dt: float,
    log_dt: float,
    seed: int | None,
) -> Tuple[Path, Path]:
    """Run case 2 TrajOpt, execute the plan, and save project-format CSV logs."""
    planned_q_path = plan_trajopt_path(render_planning=False, seed=seed)

    env, robot, _ = build_env(render=render)
    env.reset()
    safe_set_joint_angles(robot, START_JOINTS)

    times, ee_data, joint_data = execute_planned_trajectory(
        robot=robot,
        sim=robot.sim,
        planned_q_path=planned_q_path,
        source_dt=source_dt,
        log_dt=log_dt,
        render=render,
    )

    ee_file, joint_file = build_output_paths(case_id, algo_name, episode_tag, output_dir)
    save_ee_csv(ee_file, times, ee_data)
    save_joint_csv(joint_file, times, joint_data)

    final_ee_pos = robot.get_ee_position().copy()
    final_distance = np.linalg.norm(final_ee_pos - GOAL_POS)
    print(f"Final EE position: {final_ee_pos}")
    print(f"Distance to goal: {final_distance:.6f} m")
    print(f"Saved EE log    -> {ee_file}")
    print(f"Saved joint log -> {joint_file}")

    env.close()
    return ee_file, joint_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a joint-space TrajOpt planner for Panda case 2.")
    parser.add_argument("--render", action="store_true", help="Open the PyBullet GUI during execution.")
    parser.add_argument("--case-id", type=int, default=CASE_ID, help="Case identifier used in output file names.")
    parser.add_argument(
        "--algo-name",
        type=str,
        default=DEFAULT_ALGO_NAME,
        help="Algorithm name used in output file names.",
    )
    parser.add_argument(
        "--episode-tag",
        type=str,
        default=DEFAULT_EPISODE_TAG,
        help="Episode tag used in output file names.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory where EE and joint CSV logs will be saved.",
    )
    parser.add_argument(
        "--source-dt",
        type=float,
        default=DEFAULT_SOURCE_DT,
        help="Simulation step used during trajectory execution before log resampling.",
    )
    parser.add_argument(
        "--log-dt",
        type=float,
        default=DEFAULT_LOG_DT,
        help="Uniform time step used in the saved CSV logs.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    args = parser.parse_args()

    run_trajopt_case2(
        render=args.render,
        case_id=args.case_id,
        algo_name=args.algo_name,
        episode_tag=args.episode_tag,
        output_dir=Path(args.output_dir),
        source_dt=args.source_dt,
        log_dt=args.log_dt,
        seed=args.seed,
    )

