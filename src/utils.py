import numpy as np
import matplotlib.pyplot as plt

from typing import List
from gtsam import Point3, Pose3, Rot3

_M_PI_2 = np.pi / 2
_M_PI_4 = np.pi / 4


def plotPath(
    trajectory,
    orientations,
    ax,
    label="Path",
    color="blue",
    linewidth=2,
    axis_length=0.1,
    axis_step=1,
    no_legend=False,
):
    """
    Plots a 3D trajectory with orientation axes.
    """
    # Plot trajectory
    ax.plot(
        trajectory[:, 0],
        trajectory[:, 1],
        trajectory[:, 2],
        label=label,
        color=color,
        linewidth=linewidth,
    )

    # Plot orientation axes
    for i in range(0, len(trajectory), axis_step):
        pos = trajectory[i]
        R = orientations[i]  # 3x3 rotation matrix

        # Plot local X, Y, Z axes
        for j, color in enumerate(["r", "g", "b"]):  # X=Red, Y=Green, Z=Blue
            axis_vector = R[:, j] * axis_length
            ax.quiver(
                pos[0],
                pos[1],
                pos[2],
                axis_vector[0],
                axis_vector[1],
                axis_vector[2],
                color=color,
            )

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("3D Trajectory with Orientation Axes")
    if not (no_legend):
        ax.legend()
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio


def createPoses(
    init: Pose3 = Pose3(Rot3.Ypr(_M_PI_2, 0, -_M_PI_2), Point3(30, 0, 0)),
    delta: Pose3 = Pose3(
        Rot3.Ypr(0, -_M_PI_4, 0),
        Point3(np.sin(_M_PI_4) * 30, 0, 30 * (1 - np.sin(_M_PI_4))),
    ),
    steps: int = 8,
) -> List[Pose3]:
    """
    Create a set of ground-truth poses
    Default values give a circular trajectory, radius 30 at pi/4 intervals,
    always facing the circle center
    """
    poses = [init]
    for _ in range(1, steps):
        poses.append(poses[-1].compose(delta))
    return poses


def posesOnCircle(num_cameras=8, R=30):
    """Create regularly spaced poses with specified radius and number of cameras."""
    theta = 2 * np.pi / num_cameras

    # Initial pose at angle 0, position (R, 0, 0), facing the center with Y-axis pointing down
    init_rotation = Rot3.Ypr(_M_PI_2, 0, -_M_PI_2)
    init_position = np.array([R, 0, 0])
    init = Pose3(init_rotation, init_position)

    # Delta rotation: rotate by -theta around Z-axis (counterclockwise movement)
    delta_rotation = Rot3.Ypr(0, -theta, 0)

    # Delta translation in world frame
    delta_translation_world = np.array([R * (np.cos(theta) - 1), R * np.sin(theta), 0])

    # Transform delta translation to local frame of the camera
    delta_translation_local = init.rotation().unrotate(delta_translation_world)

    # Define delta pose
    delta = Pose3(delta_rotation, delta_translation_local)

    # Generate poses
    return createPoses(init, delta, num_cameras)
