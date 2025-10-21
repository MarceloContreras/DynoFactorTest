import gtsam
import numpy as np

from typing import Optional, List
from gtsam.utils.numerical_derivative import (
    numericalDerivative31,
    numericalDerivative32,
    numericalDerivative33,
    numericalDerivative41,
    numericalDerivative42,
    numericalDerivative43,
    numericalDerivative44,
)


def error_reprojection(
    measurement_K: list,
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    pose_key, point_key = this.keys()

    pose = values.atPose3(pose_key)
    point = values.atPoint3(point_key)

    # Create camera and project
    K = measurement_K[1]
    camera = gtsam.PinholeCameraCal3_S2(pose, K)
    proj = camera.project(point)
    error = proj - measurement_K[0]

    P_c = pose.transformTo(point)
    R_t = np.transpose(pose.matrix()[:3, :3])

    if jacobians is not None:
        x, y, z = P_c[0], P_c[1], P_c[2]
        dpi_dPc = (
            1
            / z
            * np.array([[K.fx(), 0, -x / z * K.fx()], [0, K.fy(), -y / z * K.fy()]])
        )
        dPc_dT = np.array(
            [[0, -z, y, -1, 0, 0], [z, 0, -x, 0, -1, 0], [-y, x, 0, 0, 0, -1]]
        )

        jacobians[0] = dpi_dPc @ dPc_dT
        jacobians[1] = dpi_dPc @ R_t

    return error


def error_object_motion(
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    def h(
        object_motion: gtsam.Pose3, prev_point: gtsam.Point3, curr_point: gtsam.Point3
    ):
        return curr_point - object_motion.transformTo(prev_point)

    motion_key, point1_key, point2_key = this.keys()

    motion12 = values.atPose3(motion_key)
    point1 = values.atPoint3(point1_key)
    point2 = values.atPoint3(point2_key)

    error = h(motion12, point1, point2)

    if jacobians is not None:
        jacobians[0] = numericalDerivative31(h, motion12, point1, point2)
        jacobians[1] = numericalDerivative32(h, motion12, point1, point2)
        jacobians[2] = numericalDerivative33(h, motion12, point1, point2)

    return error


def error_object_pose(
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    def h(
        prev_pose: gtsam.Pose3,
        curr_pose: gtsam.Pose3,
        prev_point: gtsam.Point3,
        curr_points: gtsam.Point3,
    ):
        return curr_points - (curr_pose * prev_pose.inverse()).transformTo(prev_point)

    pose1_key, pose2_key, point1_key, point2_key = this.keys()

    pose1 = values.atPose3(pose1_key)
    pose2 = values.atPose3(pose2_key)
    point1 = values.atPoint3(point1_key)
    point2 = values.atPoint3(point2_key)

    error = h(pose1, pose2, point1, point2)

    if jacobians is not None:
        jacobians[0] = numericalDerivative41(h, pose1, pose2, point1, point2)
        jacobians[1] = numericalDerivative42(h, pose1, pose2, point1, point2)
        jacobians[2] = numericalDerivative43(h, pose1, pose2, point1, point2)
        jacobians[3] = numericalDerivative44(h, pose1, pose2, point1, point2)

    return error


def error_object_pose_smoother(
    this: gtsam.CustomFactor,
    values: gtsam.Values,
    jacobians: Optional[List[np.ndarray]],
) -> np.ndarray:
    def h(pose1: gtsam.Pose3, pose2: gtsam.Pose3, pose3: gtsam.Pose3):
        k_1_H_k = pose2 * pose1.inverse()
        k_2_H_k_1 = pose3 * pose2.inverse()
        hx = gtsam.Pose3.between(k_1_H_k, k_2_H_k_1)
        I = gtsam.Pose3.Identity()
        return gtsam.Pose3.localCoordinates(I, hx)

    pose1_key, pose2_key, pose3_key = this.keys()

    pose1 = values.atPose3(pose1_key)
    pose2 = values.atPose3(pose2_key)
    pose3 = values.atPose3(pose3_key)

    error = h(pose1, pose2, pose3)

    if jacobians is not None:
        jacobians[0] = numericalDerivative31(h, pose1, pose2, pose3)
        jacobians[1] = numericalDerivative32(h, pose1, pose2, pose3)
        jacobians[2] = numericalDerivative33(h, pose1, pose2, pose3)

    return error
