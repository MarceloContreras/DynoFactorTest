import gtsam
import numpy as np
import matplotlib.pyplot as plt

from typing import Iterable
from gtsam.utils import plot
from gtsam import Point3, Values, Marginals


def plot_pose3_on_axes(
    axes, pose, axis_length=0.1, P=None, scale=1, ground_truth=False
):
    """
    Plot a 3D pose on given axis `axes` with given `axis_length`.

    The uncertainty ellipse (if covariance is given) is scaled in such a way
    that 95% of drawn samples are inliers, see `plot_covariance_ellipse_3d`.

    Args:
        axes (matplotlib.axes.Axes): Matplotlib axes.
        point (gtsam.Point3): The point to be plotted.
        linespec (string): String representing formatting options for Matplotlib.
        P (numpy.ndarray): Marginal covariance matrix to plot the uncertainty of the estimation.
    """
    # get rotation and translation (center)
    gRp = pose.rotation().matrix()  # rotation from pose to global
    origin = pose.translation()

    # draw the camera axes
    x_axis = origin + gRp[:, 0] * axis_length
    line = np.append(origin[np.newaxis], x_axis[np.newaxis], axis=0)
    if ground_truth:
        axes.plot(line[:, 0], line[:, 1], line[:, 2], "k--")
    else:
        axes.plot(line[:, 0], line[:, 1], line[:, 2], "r-")

    y_axis = origin + gRp[:, 1] * axis_length
    line = np.append(origin[np.newaxis], y_axis[np.newaxis], axis=0)
    if ground_truth:
        axes.plot(line[:, 0], line[:, 1], line[:, 2], "k--")
    else:
        axes.plot(line[:, 0], line[:, 1], line[:, 2], "g-")

    z_axis = origin + gRp[:, 2] * axis_length
    line = np.append(origin[np.newaxis], z_axis[np.newaxis], axis=0)
    if ground_truth:
        axes.plot(line[:, 0], line[:, 1], line[:, 2], "k--")
    else:
        axes.plot(line[:, 0], line[:, 1], line[:, 2], "b-")

    # plot the covariance
    if P is not None:
        # covariance matrix in pose coordinate frame
        pPp = P[3:6, 3:6]
        # convert the covariance matrix to global coordinate frame
        gPp = gRp @ pPp @ gRp.T
        plot.plot_covariance_ellipse_3d(axes, origin, gPp)


def plot_trajectory_camera(
    fignum: int,
    values: Values,
    scale: float = 1,
    marginals: Marginals = None,
    ground_truth: bool = False,
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
            plot_pose3_on_axes(
                axes, pose, P=covariance, axis_length=scale, ground_truth=ground_truth
            )

    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title.lower())


def plot_trajectory_car_per_motion(
    fignum: int,
    initial_pose: gtsam.Pose3,
    motion_values: Values,
    scale: float = 1,
    marginals: Marginals = None,
    apply_inverse: bool = True,
    ground_truth: bool = False,
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
    plot_pose3_on_axes(
        axes, pose, P=covariance, axis_length=scale, ground_truth=ground_truth
    )

    motions = gtsam.utilities.allPose3s(motion_values)
    for key in motions.keys():
        if motion_symbol in str(gtsam.Symbol(key)):
            H = motions.atPose3(key)
            if apply_inverse:
                pose = H.inverse().compose(pose)
            else:
                pose = H.compose(pose)
            if marginals:
                covariance = marginals.marginalCovariance(key)
            else:
                covariance = None
            plot_pose3_on_axes(
                axes, pose, P=covariance, axis_length=scale, ground_truth=ground_truth
            )

    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title.lower())


def plot_3d_points_car(
    fignum,
    points: np.ndarray,
    linespec="g*",
    ground_truth: bool = False,
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


def plot_results_vs_gt(
    values: Values,
    ground_truth_values: Values,
    pose_symbol: str = "x",
    title: str = "Plot Trajectory",
) -> None:

    fig, ax = plt.subplots(2, 2, sharey=True, constrained_layout=True)

    ape_x, ape_y, ape_z = [], [], []
    ape_rot_x, ape_rot_y, ape_rot_z = [], [], []
    rpe_x, rpe_y, rpe_z = [], [], []
    rpe_rot_x, rpe_rot_y, rpe_rot_z = [], [], []

    # Results
    poses, gt_poses = [], []
    poses_gtsam = gtsam.utilities.allPose3s(values)
    gt_poses_gtsam = gtsam.utilities.allPose3s(ground_truth_values)
    for key, key_gt in zip(poses_gtsam.keys(), gt_poses_gtsam.keys()):
        # Just extract camera
        if pose_symbol in str(gtsam.Symbol(key)) and pose_symbol in str(
            gtsam.Symbol(key_gt)
        ):
            poses.append(poses_gtsam.atPose3(key))
            gt_poses.append(gt_poses_gtsam.atPose3(key_gt))

    for pose, gt_pose in zip(poses, gt_poses):
        # Error computation
        ape = pose.inverse().compose(gt_pose)
        trans_ape = np.abs(ape.translation())
        rot_ape = np.abs(gtsam.Pose3.Logmap(ape)[:3])
        # Logging
        ape_x.append(trans_ape[0])
        ape_y.append(trans_ape[1])
        ape_z.append(trans_ape[2])
        ape_rot_x.append(rot_ape[0])
        ape_rot_y.append(rot_ape[1])
        ape_rot_z.append(rot_ape[2])

    for i in range(0, len(poses) - 1):
        # Error computation
        relative_gt = gt_poses[i].inverse().compose(gt_poses[i + 1])
        relative_est = poses[i].inverse().compose(poses[i + 1])
        rpe = relative_gt.inverse().compose(relative_est)
        trans_rpe = np.abs(rpe.translation())
        rot_rpe = np.abs(gtsam.Pose3.Logmap(rpe)[:3])
        # Logging
        rpe_x.append(trans_rpe[0])
        rpe_y.append(trans_rpe[1])
        rpe_z.append(trans_rpe[2])
        rpe_rot_x.append(rot_rpe[0])
        rpe_rot_y.append(rot_rpe[1])
        rpe_rot_z.append(rot_rpe[2])

    t = np.arange(len(ape_x))
    t_rpe = np.arange(len(rpe_x))

    ax[0, 0].set_title("Absolute Error")
    ax[0, 0].plot(t, ape_x, label="x")
    ax[0, 0].plot(t, ape_y, label="y")
    ax[0, 0].plot(t, ape_z, label="z")
    ax[0, 0].legend(loc="upper right")
    ax[0, 0].set_ylabel(f"$APE_{{t}}[m]$")
    ax[0, 0].set_xlabel("Frame index[#]")
    ax[0, 0].grid()

    ax[1, 0].plot(t, ape_rot_x, label="roll")
    ax[1, 0].plot(t, ape_rot_y, label="pitch")
    ax[1, 0].plot(t, ape_rot_z, label="yaw")
    ax[1, 0].legend(loc="upper right")
    ax[1, 0].set_ylabel(f"$APE_{{r}}[rad]$")
    ax[1, 0].set_xlabel("Frame index[#]")
    ax[1, 0].grid()

    ax[0, 1].set_title("Relative Error")
    ax[0, 1].plot(t_rpe, rpe_x, label="x")
    ax[0, 1].plot(t_rpe, rpe_y, label="y")
    ax[0, 1].plot(t_rpe, rpe_z, label="z")
    ax[0, 1].legend(loc="upper right")
    ax[0, 1].set_ylabel(f"$RPE_{{t}}[m]$")
    ax[0, 1].set_xlabel("Frame index[#]")
    ax[0, 1].grid()

    ax[1, 1].plot(t_rpe, rpe_rot_x, label="roll")
    ax[1, 1].plot(t_rpe, rpe_rot_y, label="pitch")
    ax[1, 1].plot(t_rpe, rpe_rot_z, label="yaw")
    ax[1, 1].legend(loc="upper right")
    ax[1, 1].set_ylabel(f"$RPE_{{r}}[rad]$")
    ax[1, 1].set_xlabel("Frame index[#]")
    ax[1, 1].grid()

    fig.suptitle(title)
    fig.canvas.manager.set_window_title(title.lower())
