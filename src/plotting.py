import gtsam
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable
from gtsam.utils import plot
from gtsam import Point3, Values, Marginals


def plot_trajectory_camera(
    fignum: int,
    values: Values,
    scale: float = 1,
    marginals: Marginals = None,
    pose_symbol: str = "x",
    title: str = "Plot Trajectory",
    axis_labels: Iterable[str] = ("X axis", "Y axis", "Z axis"),
) -> None:

    fig = plt.figure(fignum)
    if not fig.axes:
        axes = fig.add_subplot(projection="3d")
    else:
        axes = fig.axes[0]

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])

    # Then 3D poses, if any
    poses = gtsam.utilities.allPose3s(values)
    for key in poses.keys():
        if pose_symbol in str(gtsam.Symbol(key)):
            pose = poses.atPose3(key)
            if marginals:
                covariance = marginals.marginalCovariance(key)
            else:
                covariance = None
            plot.plot_pose3_on_axes(axes, pose, P=covariance, axis_length=scale)

    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title.lower())


def plot_trajectory_car_per_motion(
    fignum: int,
    initial_pose: gtsam.Pose3,
    motion_values: Values,
    scale: float = 1,
    marginals: Marginals = None,
    motion_symbol: str = "h",
    title: str = "Plot Trajectory",
    axis_labels: Iterable[str] = ("X axis", "Y axis", "Z axis"),
) -> None:

    fig = plt.figure(fignum)
    if not fig.axes:
        axes = fig.add_subplot(projection="3d")
    else:
        axes = fig.axes[0]

    axes.set_xlabel(axis_labels[0])
    axes.set_ylabel(axis_labels[1])
    axes.set_zlabel(axis_labels[2])

    # Then 3D poses, if any
    covariance = None
    pose = initial_pose
    plot.plot_pose3_on_axes(axes, pose, P=covariance, axis_length=scale)

    motions = gtsam.utilities.allPose3s(motion_values)
    for key in motions.keys():
        if motion_symbol in str(gtsam.Symbol(key)):
            H = motions.atPose3(key)
            pose = H.inverse().compose(pose)
            if marginals:
                covariance = marginals.marginalCovariance(key)
            else:
                covariance = None
            plot.plot_pose3_on_axes(axes, pose, P=covariance, axis_length=scale)

    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title.lower())


def plot_3d_points_car(
    fignum,
    points: np.ndarray,
    linespec="g*",
    marginals=None,
    title="3D Points",
    axis_labels=("X axis", "Y axis", "Z axis"),
):

    # Plot points and covariance matrices
    timestamp = points.shape[0]

    for t in range(timestamp):
        for pt in points[t]:
            fig = plot.plot_point3(
                fignum, Point3(pt), linespec, marginals, axis_labels=axis_labels
            )

    fig = plt.figure(fignum)
    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title.lower())


def plot_hessian_matrix(graph):
    keys = set()
    for i in range(graph.size()):
        factor = graph.at(i)
        for k in factor.keys():
            keys.add(k)
    keys = sorted(keys)

    # Build mapping key -> index
    key_to_idx = {k: i for i, k in enumerate(keys)}

    # Initialize adjacency matrix
    A = np.ones((len(keys), len(keys)), dtype=int)

    # Fill adjacency by checking which variables appear together in a factor
    for i in range(graph.size()):
        factor = graph.at(i)
        f_keys = list(factor.keys())
        for a in range(len(f_keys)):
            for b in range(a + 1, len(f_keys)):
                ia = key_to_idx[f_keys[a]]
                ib = key_to_idx[f_keys[b]]
                A[ia, ib] = 0
                A[ib, ia] = 0

    np.fill_diagonal(A, 0)

    plt.figure(figsize=(4, 4))
    plt.imshow(A, cmap="gray", interpolation="none")
    plt.xticks(
        range(len(keys)), [gtsam.DefaultKeyFormatter(k) for k in keys], rotation=45
    )
    plt.yticks(range(len(keys)), [gtsam.DefaultKeyFormatter(k) for k in keys])
    plt.title("Variable Connectivity Matrix (Reduced)")
    plt.tight_layout()
    plt.show()
