from __future__ import annotations

# Standard library imports.
import re
from pathlib import Path
from typing import Sequence

# Third-party imports for numerical processing, CSV loading, and plotting.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


# Expected filename format for all trajectory logs used by this script.
# The compare utilities rely on this naming convention to recover metadata such
# as case ID, algorithm name, episode tag, and whether the file contains EE or
# joint logs.
FILENAME_RE = re.compile(
    r"^case_(?P<case_id>\d+)_(?P<algo>[A-Za-z0-9_]+)_(?P<episode_tag>.+)_(?P<kind>ee|joints)\.csv$"
)

# Fixed color palette for the seven Panda arm joints. Keeping the mapping stable
# across figures makes visual comparison between algorithms easier.
JOINT_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "saddlebrown",
    "hotpink",
]


def parse_log_filename(csv_path: str | Path) -> dict:
    """Extract metadata from a standardized CSV filename."""
    path = Path(csv_path)
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(
            "CSV filename must match 'case_<id>_<algo>_<episode_tag>_<ee|joints>.csv'. "
            f"Got: {path.name}"
        )
    meta = match.groupdict()
    meta["case_id"] = int(meta["case_id"])
    meta["path"] = path
    # This prefix can be useful when EE and joint logs belong to the same run.
    meta["stem_without_kind"] = path.stem.rsplit("_", 1)[0]
    return meta


# -----------------------------------------------------------------------------
# CSV loading helpers
# -----------------------------------------------------------------------------
# These functions validate the log schema before the plotting/metric code runs.
# That prevents hard-to-read downstream errors when a CSV has a wrong format.
def load_ee_csv(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    meta = parse_log_filename(csv_path)
    df = pd.read_csv(csv_path)
    expected = ["timestep", "time_sec", "posx", "posy", "posz", "velx", "vely", "velz"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing EE columns in {csv_path}: {missing}")
    return df.copy(), meta



def load_joints_csv(csv_path: str | Path) -> tuple[pd.DataFrame, dict]:
    meta = parse_log_filename(csv_path)
    df = pd.read_csv(csv_path)
    expected = ["timestep", "time_sec", "j1", "j2", "j3", "j4", "j5", "j6", "j7"]
    missing = [col for col in expected if col not in df.columns]
    if missing:
        raise ValueError(f"Missing joint columns in {csv_path}: {missing}")
    return df.copy(), meta


# -----------------------------------------------------------------------------
# Derivative and scalar metric helpers
# -----------------------------------------------------------------------------
# The compare script derives velocities/accelerations from time series data
# stored in the CSV logs. All derivatives are computed against time_sec to avoid
# assuming a hard-coded sampling interval.
def compute_time_derivative(time_sec: np.ndarray, values: np.ndarray) -> np.ndarray:
    time_sec = np.asarray(time_sec, dtype=float)
    values = np.asarray(values, dtype=float)

    # Normalize 1D input into a column vector so the same implementation works
    # both for scalar time series and multi-axis signals.
    if values.ndim == 1:
        values = values[:, None]

    if len(time_sec) != values.shape[0]:
        raise ValueError("time_sec length must match the number of rows in values")
    if len(time_sec) < 2:
        return np.zeros_like(values, dtype=float)
    if np.any(np.diff(time_sec) <= 0):
        raise ValueError("time_sec must be strictly increasing")

    # np.gradient gives a compact way to compute numerical derivatives while
    # handling edge samples without explicit manual branching.
    return np.gradient(values, time_sec, axis=0, edge_order=1)



def compute_joint_velocity_acceleration(joints_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    time_sec = joints_df["time_sec"].to_numpy(dtype=float)
    joint_cols = [f"j{i}" for i in range(1, 8)]
    joint_values = joints_df[joint_cols].to_numpy(dtype=float)

    # First derivative: joint velocity. Second derivative: joint acceleration.
    joint_vel = compute_time_derivative(time_sec, joint_values)
    joint_acc = compute_time_derivative(time_sec, joint_vel)

    vel_df = pd.DataFrame(joint_vel, columns=[f"dj{i}" for i in range(1, 8)])
    acc_df = pd.DataFrame(joint_acc, columns=[f"ddj{i}" for i in range(1, 8)])
    vel_df.insert(0, "time_sec", time_sec)
    acc_df.insert(0, "time_sec", time_sec)
    return vel_df, acc_df



def compute_ee_acceleration(ee_df: pd.DataFrame) -> pd.DataFrame:
    time_sec = ee_df["time_sec"].to_numpy(dtype=float)
    vel_values = ee_df[["velx", "vely", "velz"]].to_numpy(dtype=float)
    acc_values = compute_time_derivative(time_sec, vel_values)

    acc_df = pd.DataFrame(acc_values, columns=["accx", "accy", "accz"])
    acc_df.insert(0, "time_sec", time_sec)
    return acc_df



def compute_ee_motion_velocity_acceleration_scalar(ee_df: pd.DataFrame) -> pd.DataFrame:
    time_sec = ee_df["time_sec"].to_numpy(dtype=float)
    pos = ee_df[["posx", "posy", "posz"]].to_numpy(dtype=float)
    vel = ee_df[["velx", "vely", "velz"]].to_numpy(dtype=float)
    acc = compute_ee_acceleration(ee_df)[["accx", "accy", "accz"]].to_numpy(dtype=float)

    # Motion is measured as displacement magnitude from the initial EE position.
    motion = np.linalg.norm(pos - pos[0], axis=1)
    # Speed and acceleration are reduced to scalar magnitudes for one combined plot.
    speed = np.linalg.norm(vel, axis=1)
    acceleration = np.linalg.norm(acc, axis=1)

    return pd.DataFrame(
        {
            "time_sec": time_sec,
            "motion": motion,
            "speed": speed,
            "acceleration": acceleration,
        }
    )



def compute_cartesian_distance(ee_csv_path: str | Path) -> float:
    """
    Compute the Cartesian distance criterion as the sum of Euclidean distances
    between consecutive end-effector positions.
    """
    ee_df, _ = load_ee_csv(ee_csv_path)
    pos = ee_df[["posx", "posy", "posz"]].to_numpy(dtype=float)
    if len(pos) < 2:
        return 0.0
    step_vectors = np.diff(pos, axis=0)
    return float(np.linalg.norm(step_vectors, axis=1).sum())



def compute_joint_distance(joints_csv_path: str | Path) -> float:
    """
    Compute the joint distance criterion as the accumulated absolute joint change
    over all consecutive waypoints.
    """
    joints_df, _ = load_joints_csv(joints_csv_path)
    joints = joints_df[[f"j{i}" for i in range(1, 8)]].to_numpy(dtype=float)
    if len(joints) < 2:
        return 0.0
    joint_steps = np.diff(joints, axis=0)
    return float(np.abs(joint_steps).sum())



def print_distance_criteria(
    ee_csv_path: str | Path,
    joints_csv_path: str | Path,
) -> tuple[float, float]:
    """Print and return Cartesian distance and joint distance for one trajectory."""
    cartesian_distance = compute_cartesian_distance(ee_csv_path)
    joint_distance = compute_joint_distance(joints_csv_path)
    print(f"Cartesian distance [m]: {cartesian_distance:.6f}")
    print(f"Joint distance [rad]: {joint_distance:.6f}")
    return cartesian_distance, joint_distance


# -----------------------------------------------------------------------------
# Scene helpers for 3D overlays
# -----------------------------------------------------------------------------
# The 3D comparison plots use a simplified analytical description of each case:
# goal position, obstacle geometry, and axis limits. The trajectories come from
# CSV logs; the scene is reconstructed here only for visualization.
def _get_case_scene(case_id: int) -> dict:
    if case_id == 0:
        return {
            "title": "Case 0",
            "goal": np.array([0.4, 0.0, 0.1], dtype=float),
            "walls": [],
            "xlim": (0.0, 0.65),
            "ylim": (-0.45, 0.45),
            "zlim": (0.0, 0.55),
        }
    if case_id == 1:
        return {
            "title": "Case 1",
            "goal": np.array([0.4, 0.0, 0.1], dtype=float),
            "walls": [
                {
                    "x_min": 0.20 - 0.01 / 2.0,
                    "x_max": 0.20 + 0.01 / 2.0,
                    "y_min": -0.2,
                    "y_max": 0.2,
                    "z_min": 0.0,
                    "z_max": 0.4,
                    "name": "Small wall",
                }
            ],
            "xlim": (0.0, 0.65),
            "ylim": (-0.45, 0.45),
            "zlim": (0.0, 0.55),
        }
    if case_id == 2:
        return {
            "title": "Case 2",
            "goal": np.array([0.4, 0.0, 0.05], dtype=float),
            "walls": [
                {
                    "x_min": 0.35 - 0.02 / 2.0,
                    "x_max": 0.35 + 0.02 / 2.0,
                    "y_min": -1.5 / 2.0,
                    "y_max": 1.5 / 2.0,
                    "z_min": 0.30,
                    "z_max": 0.80,
                    "name": "High wall",
                }
            ],
            "xlim": (0.0, 0.65),
            "ylim": (-0.8, 0.8),
            "zlim": (0.0, 0.9),
        }
    raise ValueError(f"Unsupported case_id: {case_id}")



def _make_box_faces(box: dict) -> list[list[list[float]]]:
    x0, x1 = box["x_min"], box["x_max"]
    y0, y1 = box["y_min"], box["y_max"]
    z0, z1 = box["z_min"], box["z_max"]

    # Eight box vertices.
    v000 = [x0, y0, z0]
    v100 = [x1, y0, z0]
    v110 = [x1, y1, z0]
    v010 = [x0, y1, z0]
    v001 = [x0, y0, z1]
    v101 = [x1, y0, z1]
    v111 = [x1, y1, z1]
    v011 = [x0, y1, z1]

    # Six quad faces for Poly3DCollection.
    return [
        [v000, v100, v110, v010],
        [v001, v101, v111, v011],
        [v000, v100, v101, v001],
        [v010, v110, v111, v011],
        [v000, v010, v011, v001],
        [v100, v110, v111, v101],
    ]



def _create_figure(figsize: tuple[float, float] = (10.0, 6.0)):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax



def _build_3d_axes(case_id: int, title: str):
    scene = _get_case_scene(case_id)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Draw all walls for the selected case as translucent boxes.
    for wall in scene["walls"]:
        faces = _make_box_faces(wall)
        wall_poly = Poly3DCollection(faces, alpha=0.25, facecolor="salmon", edgecolor="none")
        ax.add_collection3d(wall_poly)

    # Goal is shown explicitly to make success/failure visually obvious.
    goal = scene["goal"]
    ax.scatter(goal[0], goal[1], goal[2], color="magenta", s=70, label="Goal")

    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_xlim(*scene["xlim"])
    ax.set_ylim(*scene["ylim"])
    ax.set_zlim(*scene["zlim"])
    return fig, ax



def _save_figure(fig, output_path: str | Path | None) -> None:
    if output_path is None:
        return
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")


# -----------------------------------------------------------------------------
# Plotting functions
# -----------------------------------------------------------------------------
# All plotting functions follow the same pattern:
# 1. Load and validate the CSV.
# 2. Build the appropriate plot.
# 3. Optionally save it.
# 4. Optionally display it interactively.
def plot_same_algorithm_trajectories_3d(
    ee_csv_paths: Sequence[str | Path],
    case_id: int | None = None,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    if not ee_csv_paths:
        raise ValueError("ee_csv_paths cannot be empty")

    # The first file defines the default algorithm name and, unless overridden,
    # the case ID used to construct the scene.
    first_df, first_meta = load_ee_csv(ee_csv_paths[0])
    resolved_case_id = first_meta["case_id"] if case_id is None else case_id
    algo_name = first_meta["algo"]

    fig, ax = _build_3d_axes(
        case_id=resolved_case_id,
        title=title or f"{algo_name.upper()} trajectories overlay — case {resolved_case_id}",
    )

    for idx, path in enumerate(ee_csv_paths):
        df, meta = load_ee_csv(path)
        ax.plot(df["posx"], df["posy"], df["posz"], linewidth=2.0, label=meta["episode_tag"])
        # Mark the start point once to avoid duplicate legend entries.
        if idx == 0:
            ax.scatter(df["posx"].iloc[0], df["posy"].iloc[0], df["posz"].iloc[0], color="green", s=40, label="Start")

    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_multi_algorithm_trajectories_3d(
    ee_csv_paths: Sequence[str | Path],
    case_id: int,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    if not ee_csv_paths:
        raise ValueError("ee_csv_paths cannot be empty")

    fig, ax = _build_3d_axes(case_id=case_id, title=title or f"Algorithm comparison — case {case_id}")

    for idx, path in enumerate(ee_csv_paths):
        df, meta = load_ee_csv(path)
        ax.plot(df["posx"], df["posy"], df["posz"], linewidth=2.2, label=meta["algo"].upper())
        # Mark the common start location using the first trajectory only.
        if idx == 0:
            ax.scatter(df["posx"].iloc[0], df["posy"].iloc[0], df["posz"].iloc[0], color="green", s=40, label="Start")

    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_joint_angles_vs_time(
    joints_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_joints_csv(joints_csv_path)
    fig, ax = _create_figure()

    for i in range(1, 8):
        ax.plot(df["time_sec"], df[f"j{i}"], label=f"j{i}", color=JOINT_COLORS[i - 1], linewidth=1.8)

    ax.set_title(title or f"{meta['algo'].upper()} joint angles vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint angle [rad]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_joint_velocities_vs_time(
    joints_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_joints_csv(joints_csv_path)
    vel_df, _ = compute_joint_velocity_acceleration(df)
    fig, ax = _create_figure()

    for i in range(1, 8):
        ax.plot(vel_df["time_sec"], vel_df[f"dj{i}"], label=f"dj{i}", color=JOINT_COLORS[i - 1], linewidth=1.8)

    ax.set_title(title or f"{meta['algo'].upper()} joint velocities vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint velocity [rad/s]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_joint_accelerations_vs_time(
    joints_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_joints_csv(joints_csv_path)
    _, acc_df = compute_joint_velocity_acceleration(df)
    fig, ax = _create_figure()

    for i in range(1, 8):
        ax.plot(acc_df["time_sec"], acc_df[f"ddj{i}"], label=f"ddj{i}", color=JOINT_COLORS[i - 1], linewidth=1.8)

    ax.set_title(title or f"{meta['algo'].upper()} joint accelerations vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint acceleration [rad/s²]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_ee_position_vs_time(
    ee_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_ee_csv(ee_csv_path)
    fig, ax = _create_figure()

    ax.plot(df["time_sec"], df["posx"], label="x", linewidth=2.0)
    ax.plot(df["time_sec"], df["posy"], label="y", linewidth=2.0)
    ax.plot(df["time_sec"], df["posz"], label="z", linewidth=2.0)

    ax.set_title(title or f"{meta['algo'].upper()} EE position vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Position [m]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_ee_velocity_vs_time(
    ee_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_ee_csv(ee_csv_path)
    fig, ax = _create_figure()

    ax.plot(df["time_sec"], df["velx"], label="vx", linewidth=2.0)
    ax.plot(df["time_sec"], df["vely"], label="vy", linewidth=2.0)
    ax.plot(df["time_sec"], df["velz"], label="vz", linewidth=2.0)

    ax.set_title(title or f"{meta['algo'].upper()} EE velocity vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Velocity [m/s]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_ee_acceleration_vs_time(
    ee_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_ee_csv(ee_csv_path)
    acc_df = compute_ee_acceleration(df)
    fig, ax = _create_figure()

    ax.plot(acc_df["time_sec"], acc_df["accx"], label="ax", linewidth=2.0)
    ax.plot(acc_df["time_sec"], acc_df["accy"], label="ay", linewidth=2.0)
    ax.plot(acc_df["time_sec"], acc_df["accz"], label="az", linewidth=2.0)

    ax.set_title(title or f"{meta['algo'].upper()} EE acceleration vs time")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig



def plot_ee_motion_velocity_acceleration_combined(
    ee_csv_path: str | Path,
    title: str | None = None,
    output_path: str | Path | None = None,
    show: bool = True,
):
    df, meta = load_ee_csv(ee_csv_path)
    combined_df = compute_ee_motion_velocity_acceleration_scalar(df)
    fig, ax = _create_figure()

    # This plot intentionally mixes different physical quantities to provide a
    # compact high-level view of how the end-effector moved during the run.
    ax.plot(combined_df["time_sec"], combined_df["motion"], label="motion", linewidth=2.2)
    ax.plot(combined_df["time_sec"], combined_df["speed"], label="speed", linewidth=2.2)
    ax.plot(combined_df["time_sec"], combined_df["acceleration"], label="acceleration", linewidth=2.2)

    ax.set_title(title or f"{meta['algo'].upper()} EE motion + speed + acceleration")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Value [mixed units]")
    ax.grid(True, alpha=0.35)
    ax.legend()
    _save_figure(fig, output_path)
    if show:
        plt.show()
    return fig


if __name__ == "__main__":
    from pathlib import Path

    # Output directory for all generated comparison figures.
    output_dir = Path("outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    # This main block is configured for case 2. The plotting utilities above can
    # still be reused for other cases by changing the file paths and case_id.
    case_id = 2

    # Each entry groups the EE and joint logs produced by one algorithm.
    trajectories = {
        "td3": {
            "ee": Path("case_2_td3_episode_ee.csv"),
            "joints": Path("case_2_td3_episode_joints.csv"),
        },
        "sac": {
            "ee": Path("case_2_sac_episode_ee.csv"),
            "joints": Path("case_2_sac_episode_joints.csv"),
        },
        "ppo": {
            "ee": Path("case_2_ppo_episode_ee.csv"),
            "joints": Path("case_2_ppo_episode_joints.csv"),
        },
        "tqc": {
            "ee": Path("case_2_tqc_episode_ee.csv"),
            "joints": Path("case_2_tqc_episode_joints.csv"),
        },
        "rrt-connect": {
            "ee": Path("case_2_rrt_connect_episode_ee.csv"),
            "joints": Path("case_2_rrt_connect_episode_joints.csv"),
        },
        "trajopt": {
            "ee": Path("case_2_trajopt_episode_ee.csv"),
            "joints": Path("case_2_trajopt_episode_joints.csv"),
        },
    }

    # ---------------------------------------------------------
    # 1. Print Cartesian distance and Joint distance
    # ---------------------------------------------------------
    print("\n=== Distance Criteria ===")
    for algo_name, files in trajectories.items():
        print(f"\n--- {algo_name.upper()} ---")
        print_distance_criteria(files["ee"], files["joints"])

    # ---------------------------------------------------------
    # 2. Overlay all EE trajectories in one 3D plot
    # ---------------------------------------------------------
    plot_multi_algorithm_trajectories_3d(
        [files["ee"] for files in trajectories.values()],
        case_id=case_id,
        output_path=output_dir / "case_0_all_algorithms_ee_overlay.png",
    )

    # ---------------------------------------------------------
    # 3. Per-algorithm plots
    #    - joint positions only
    #    - EE motion/velocity/acceleration combined
    # ---------------------------------------------------------
    for algo_name, files in trajectories.items():
        # Joint positions j1...j7 vs time
        plot_joint_angles_vs_time(
            files["joints"],
            output_path=output_dir / f"{algo_name}_joint_positions.png",
        )

        # EE motion + speed + acceleration on one graph
        plot_ee_motion_velocity_acceleration_combined(
            files["ee"],
            output_path=output_dir / f"{algo_name}_ee_motion_velocity_acceleration.png",
        )

    print("\nDone. All requested plots and metrics have been generated.")

