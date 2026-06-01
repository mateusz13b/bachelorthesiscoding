import argparse
import csv
import time
from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from sklearn.neighbors import NearestNeighbors

#python case0_prm.py --render


CASE_ID = 0
DEFAULT_ALGO_NAME = "prm_jointspace"
DEFAULT_EPISODE_TAG = "episode"
DEFAULT_SOURCE_DT = 1.0 / 240.0
DEFAULT_LOG_DT = 0.02
START_JOINTS = np.array([2.7, 0.2, 0.2, -1.0, 0.0, 1.7, 0.7], dtype=float)
GOAL_POS = np.array([0.4, 0.0, 0.1], dtype=float)


class PandaReachCase0Task(Reach):
    """Case 0: fixed end-effector goal with no obstacles."""

    def __init__(self, sim, get_ee_position, reward_type: str = "dense"):
        super().__init__(
            sim,
            get_ee_position=get_ee_position,
            reward_type=reward_type,
            distance_threshold=0.01,
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
    """Build a case 0 environment and return environment, robot, and simulator."""
    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render_mode="human") if render else PyBullet()
    robot = CustomPandaRobot(sim)
    task = PandaReachCase0Task(sim, robot.get_ee_position)
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
        self.q_min = np.array([-3.0] * self.n_dof, dtype=float)
        self.q_max = np.array([3.0] * self.n_dof, dtype=float)

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

    def solve(self, max_iters: int = 220, alpha: float = 0.14) -> np.ndarray:
        seeds = [
            self.start_q.copy(),
            self.start_q + np.array([0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
            self.start_q + np.array([-0.40, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00]),
            self.start_q + np.array([0.25, 0.20, 0.00, 0.00, 0.00, 0.00, 0.00]),
            self.start_q + np.array([-0.25, -0.20, 0.00, 0.00, 0.00, 0.00, 0.00]),
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


class JointSpacePRMPlanner:
    """Joint-space PRM planner for case 0."""

    def __init__(
        self,
        start_q: np.ndarray,
        goal_q: np.ndarray,
        q_min: np.ndarray,
        q_max: np.ndarray,
        num_samples: int = 220,
        k_neighbors: int = 12,
        edge_substeps: int = 16,
        random_seed: int | None = None,
    ):
        self.start_q = np.array(start_q[:7], dtype=float)
        self.goal_q = np.array(goal_q[:7], dtype=float)
        self.q_min = np.array(q_min[:7], dtype=float)
        self.q_max = np.array(q_max[:7], dtype=float)
        self.num_samples = num_samples
        self.k_neighbors = k_neighbors
        self.edge_substeps = edge_substeps
        self.rng = np.random.default_rng(random_seed)

    def sample_joint_configuration(self) -> np.ndarray:
        return self.rng.uniform(self.q_min, self.q_max)

    def is_configuration_valid(self, q: np.ndarray) -> bool:
        return np.all(q >= self.q_min) and np.all(q <= self.q_max)

    def is_edge_valid(self, q_from: np.ndarray, q_to: np.ndarray) -> bool:
        for alpha in np.linspace(0.0, 1.0, self.edge_substeps):
            q = (1.0 - alpha) * q_from + alpha * q_to
            if not self.is_configuration_valid(q):
                return False
        return True

    def densify_path(self, path: np.ndarray, max_segment_norm: float = 0.22) -> np.ndarray:
        densified = [path[0]]
        for i in range(len(path) - 1):
            q_from = path[i]
            q_to = path[i + 1]
            dist = float(np.linalg.norm(q_to - q_from))
            n_segments = max(1, int(np.ceil(dist / max_segment_norm)))
            for k in range(1, n_segments + 1):
                alpha = k / n_segments
                q = (1.0 - alpha) * q_from + alpha * q_to
                densified.append(q)
        return np.asarray(densified, dtype=float)

    def plan(self) -> np.ndarray:
        samples = [self.start_q, self.goal_q]
        while len(samples) < self.num_samples:
            q = self.sample_joint_configuration()
            if self.is_configuration_valid(q):
                samples.append(q)

        samples = np.asarray(samples, dtype=float)
        graph = nx.Graph()
        knn = NearestNeighbors(n_neighbors=min(self.k_neighbors, len(samples)))
        knn.fit(samples)

        for i, q in enumerate(samples):
            distances, indices = knn.kneighbors([q])
            for dist, neighbor_idx in zip(distances[0], indices[0]):
                if i == neighbor_idx:
                    continue
                neighbor_q = samples[neighbor_idx]
                if self.is_edge_valid(q, neighbor_q):
                    graph.add_edge(i, neighbor_idx, weight=float(dist))

        try:
            path_indices = nx.shortest_path(graph, source=0, target=1, weight="weight")
        except nx.NetworkXNoPath as exc:
            raise RuntimeError("No PRM path was found in joint space.") from exc

        coarse_path = samples[path_indices]
        dense_path = self.densify_path(coarse_path, max_segment_norm=0.22)
        dense_path[0] = self.start_q
        dense_path[-1] = self.goal_q
        return smooth_trajectory(dense_path, passes=5, lam=0.18)


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
    interpolation_substeps: int = 10,
    controller_inner_steps: int = 3,
    tracking_gain: float = 0.30,
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

        for _ in range(8):
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


def plan_prm_jointspace_case0(seed: int | None = None) -> np.ndarray:
    """Plan a joint-space PRM path for case 0."""
    env, robot, _ = build_env(render=False)
    obs, _ = env.reset()
    start_q = get_current_arm_joint_angles(robot)
    goal_pos = obs["desired_goal"]

    ik_solver = GoalIKSolver(
        robot=robot,
        start_q=start_q,
        goal_pos=goal_pos,
        regularization_weight=1e-3,
    )
    goal_q = ik_solver.solve(max_iters=220, alpha=0.14)

    prm_planner = JointSpacePRMPlanner(
        start_q=start_q,
        goal_q=goal_q,
        q_min=np.array([-3.0] * 7, dtype=float),
        q_max=np.array([3.0] * 7, dtype=float),
        num_samples=220,
        k_neighbors=12,
        edge_substeps=16,
        random_seed=seed,
    )
    planned_q_path = prm_planner.plan()
    env.close()
    return planned_q_path


def run_prm_jointspace_case0(
    render: bool,
    case_id: int,
    algo_name: str,
    episode_tag: str,
    output_dir: Path,
    source_dt: float,
    log_dt: float,
    seed: int | None,
) -> Tuple[Path, Path]:
    """Run case 0 joint-space PRM, execute the path, and save project-format CSV logs."""
    planned_q_path = plan_prm_jointspace_case0(seed=seed)

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
    parser = argparse.ArgumentParser(description="Run a joint-space PRM planner for Panda case 0.")
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

    run_prm_jointspace_case0(
        render=args.render,
        case_id=args.case_id,
        algo_name=args.algo_name,
        episode_tag=args.episode_tag,
        output_dir=Path(args.output_dir),
        source_dt=args.source_dt,
        log_dt=args.log_dt,
        seed=args.seed,
    )

