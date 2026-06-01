
import argparse
import csv
from pathlib import Path

import gymnasium as gym
import numpy as np
import networkx as nx
from panda_gym.envs.core import RobotTaskEnv
from panda_gym.envs.robots.panda import Panda
from panda_gym.envs.tasks.reach import Reach
from sklearn.neighbors import NearestNeighbors


# ============================================================
#  Case 1 environment (small wall)
# ============================================================
class PandaReachWallTask(Reach):
    def __init__(self, sim, get_ee_position, reward_type="dense"):
        super().__init__(
            sim,
            get_ee_position=get_ee_position,
            reward_type=reward_type,
            distance_threshold=0.01,
        )
        self.wall_x = 0.20
        self.wall_width = 0.01
        self.wall_height = 0.2
        self.sim.create_box(
            body_name="wall",
            half_extents=[self.wall_width / 2, 0.2, self.wall_height],
            mass=0,
            position=[self.wall_x, 0.0, self.wall_height / 2],
            rgba_color=[0.8, 0.1, 0.1, 1.0],
        )

    def reset(self):
        self.goal = np.array([0.4, 0.0, 0.1])
        self.sim.set_base_pose("target", self.goal, np.array([0, 0, 0, 1]))


class CustomPandaRobot(Panda):
    def reset(self):
        super().reset()
        neutral_joint_values = np.array([2.7, 0.2, 0.2, -1.0, 0.0, 1.7, 0.7])
        self.set_joint_angles(neutral_joint_values)


# ============================================================
#  Helpers
# ============================================================
def infer_control_dt(sim, default_dt=0.04):
    """Try to infer simulator/control time step, otherwise use default_dt."""
    for attr in ["dt", "time_step", "timestep", "control_dt"]:
        if hasattr(sim, attr):
            try:
                return float(getattr(sim, attr))
            except Exception:
                pass
    return float(default_dt)


def get_current_arm_joint_angles(robot):
    """Return the 7 Panda arm joint angles in radians."""
    return np.array([robot.get_joint_angle(i) for i in range(7)], dtype=float)


def save_ee_log(ee_log, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestep", "time_sec", "posx", "posy", "posz", "velx", "vely", "velz"]
        )
        writer.writerows(ee_log)
    print(f"EE log saved to {filename}, samples: {len(ee_log)}")


def save_joint_log(joint_log, filename):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["timestep", "time_sec", "j1", "j2", "j3", "j4", "j5", "j6", "j7"]
        )
        writer.writerows(joint_log)
    print(f"Joint log saved to {filename}, samples: {len(joint_log)}")


def resample_log_rows(rows, log_dt, value_slice):
    """
    Resample rows to a uniform log_dt grid.

    rows: list-like of [timestep, time_sec, ...values]
    value_slice: slice selecting numeric values after time_sec
    """
    if not rows:
        return []

    arr = np.asarray(rows, dtype=float)
    times = arr[:, 1]
    values = arr[:, value_slice]

    if len(arr) == 1 or times[-1] <= times[0]:
        return [[0, 0.0, *values[0].tolist()]]

    n_steps = int(np.round(times[-1] / log_dt))
    target_times = np.arange(n_steps + 1, dtype=float) * log_dt
    if target_times[-1] < times[-1] - 1e-9:
        target_times = np.append(target_times, times[-1])

    resampled_values = []
    for col_idx in range(values.shape[1]):
        resampled_values.append(np.interp(target_times, times, values[:, col_idx]))
    resampled_values = np.stack(resampled_values, axis=1)

    out = []
    for i, (t, vals) in enumerate(zip(target_times, resampled_values)):
        out.append([i, float(t), *vals.tolist()])
    return out


# ============================================================
#  PRM planner in workspace
# ============================================================
class PRMPlanner:
    def __init__(self, start, goal, wall_x, num_samples=200, k_neighbors=10):
        self.start = np.asarray(start, dtype=float)
        self.goal = np.asarray(goal, dtype=float)
        self.wall_x = float(wall_x)
        self.num_samples = int(num_samples)
        self.k_neighbors = int(k_neighbors)
        self.limits = [(0.0, 0.6), (-0.4, 0.4), (0.0, 0.6)]

    def is_collision_free(self, point):
        point = np.asarray(point, dtype=float)
        if abs(point[0] - self.wall_x) < 0.03 and point[2] < 0.25:
            return False
        return True

    def plan(self):
        samples = [self.start, self.goal]
        while len(samples) < self.num_samples:
            pt = np.array([np.random.uniform(l, h) for l, h in self.limits])
            if self.is_collision_free(pt):
                samples.append(pt)

        samples = np.asarray(samples, dtype=float)

        graph = nx.Graph()
        knn = NearestNeighbors(n_neighbors=self.k_neighbors)
        knn.fit(samples)

        for i, pt in enumerate(samples):
            distances, indices = knn.kneighbors([pt])
            for dist, neighbor_idx in zip(distances[0], indices[0]):
                if i == neighbor_idx:
                    continue
                neighbor_pt = samples[neighbor_idx]
                path_segment = np.linspace(pt, neighbor_pt, 10)
                if all(self.is_collision_free(p) for p in path_segment):
                    graph.add_edge(i, neighbor_idx, weight=float(dist))

        try:
            path_indices = nx.shortest_path(graph, source=0, target=1, weight="weight")
            return samples[path_indices]
        except nx.NetworkXNoPath:
            print("No PRM path found.")
            return None


# ============================================================
#  Main execution + logging
# ============================================================
def run_prm_case1(
    render=False,
    case_id=1,
    algo_name="prm",
    episode_tag="episode",
    output_dir=".",
    log_dt=0.02,
    num_samples=200,
    k_neighbors=10,
    max_steps_per_waypoint=50,
    waypoint_tolerance=0.02,
):
    from panda_gym.pybullet import PyBullet

    sim = PyBullet(render_mode="human") if render else PyBullet()
    robot = CustomPandaRobot(sim)
    task = PandaReachWallTask(sim, robot.get_ee_position)
    env = RobotTaskEnv(robot, task)
    env = gym.wrappers.TimeLimit(env, max_episode_steps=3000)

    obs, _ = env.reset()
    start_pos = obs["achieved_goal"]
    goal_pos = obs["desired_goal"]

    control_dt = infer_control_dt(env.unwrapped.sim, default_dt=0.04)

    print("--- Planning with PRM ---")
    planner = PRMPlanner(
        start=start_pos,
        goal=goal_pos,
        wall_x=0.20,
        num_samples=num_samples,
        k_neighbors=k_neighbors,
    )
    planned_path = planner.plan()

    raw_ee_log = []
    raw_joint_log = []

    step_idx = 0
    current_pos = env.unwrapped.robot.get_ee_position().copy()
    current_q = get_current_arm_joint_angles(env.unwrapped.robot)

    raw_ee_log.append(
        [
            step_idx,
            0.0,
            current_pos[0],
            current_pos[1],
            current_pos[2],
            0.0,
            0.0,
            0.0,
        ]
    )
    raw_joint_log.append(
        [
            step_idx,
            0.0,
            current_q[0],
            current_q[1],
            current_q[2],
            current_q[3],
            current_q[4],
            current_q[5],
            current_q[6],
        ]
    )

    if planned_path is not None:
        prev_pos = current_pos.copy()

        for target_point in planned_path:
            for _ in range(max_steps_per_waypoint):
                current_pos = env.unwrapped.robot.get_ee_position().copy()
                diff = target_point - current_pos

                if np.linalg.norm(diff) < waypoint_tolerance:
                    break

                action = np.append(np.clip(diff * 10.0, -1.0, 1.0), 0.0)
                obs, _, terminated, truncated, _ = env.step(action)

                new_pos = env.unwrapped.robot.get_ee_position().copy()
                current_q = get_current_arm_joint_angles(env.unwrapped.robot)

                try:
                    ee_vel = env.unwrapped.robot.get_ee_velocity().copy()
                except Exception:
                    ee_vel = (new_pos - prev_pos) / control_dt

                step_idx += 1
                time_sec = step_idx * control_dt

                raw_ee_log.append(
                    [
                        step_idx,
                        time_sec,
                        new_pos[0],
                        new_pos[1],
                        new_pos[2],
                        ee_vel[0],
                        ee_vel[1],
                        ee_vel[2],
                    ]
                )
                raw_joint_log.append(
                    [
                        step_idx,
                        time_sec,
                        current_q[0],
                        current_q[1],
                        current_q[2],
                        current_q[3],
                        current_q[4],
                        current_q[5],
                        current_q[6],
                    ]
                )

                prev_pos = new_pos.copy()

                if terminated or truncated:
                    break

            if terminated or truncated:
                break

    ee_log = resample_log_rows(raw_ee_log, log_dt=log_dt, value_slice=slice(2, 8))
    joint_log = resample_log_rows(raw_joint_log, log_dt=log_dt, value_slice=slice(2, 9))

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ee_filename = output_path / f"case_{case_id}_{algo_name}_{episode_tag}_ee.csv"
    joint_filename = output_path / f"case_{case_id}_{algo_name}_{episode_tag}_joints.csv"

    save_ee_log(ee_log, ee_filename)
    save_joint_log(joint_log, joint_filename)

    final_ee = env.unwrapped.robot.get_ee_position().copy()
    final_q = get_current_arm_joint_angles(env.unwrapped.robot)

    print("\n--- Final robot configuration ---")
    for i, angle in enumerate(final_q, start=1):
        print(f"Joint {i}: {angle:.4f} rad")

    print(f"Final EE position: {final_ee}")
    print(f"Distance to goal: {np.linalg.norm(final_ee - goal_pos):.6f} m")

    env.close()

    return {
        "planned_path": planned_path,
        "ee_log": ee_log,
        "joint_log": joint_log,
        "goal_pos": goal_pos,
        "final_ee": final_ee,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="Open PyBullet GUI.")
    parser.add_argument("--case-id", type=int, default=1)
    parser.add_argument("--algo-name", type=str, default="prm")
    parser.add_argument("--episode-tag", type=str, default="episode")
    parser.add_argument("--output-dir", type=str, default=".")
    parser.add_argument("--log-dt", type=float, default=0.02)
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--k-neighbors", type=int, default=10)
    parser.add_argument("--max-steps-per-waypoint", type=int, default=50)
    parser.add_argument("--waypoint-tolerance", type=float, default=0.02)
    args = parser.parse_args()

    run_prm_case1(
        render=args.render,
        case_id=args.case_id,
        algo_name=args.algo_name,
        episode_tag=args.episode_tag,
        output_dir=args.output_dir,
        log_dt=args.log_dt,
        num_samples=args.num_samples,
        k_neighbors=args.k_neighbors,
        max_steps_per_waypoint=args.max_steps_per_waypoint,
        waypoint_tolerance=args.waypoint_tolerance,
    )

