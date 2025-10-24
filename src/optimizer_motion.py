import gtsam
import plotting
import numpy as np
import matplotlib.pyplot as plt

from map import Map
from gtsam.utils import plot
from gtsam import symbol_shorthand
from gtsam import (
    Cal3_S2,
    LevenbergMarquardtOptimizer,
    NonlinearFactorGraph,
    PriorFactorPoint3,
    PriorFactorPose3,
    Values,
    Point3,
    Pose3,
    Marginals,
)
from functools import partial
from factors import error_reprojection, error_object_motion

L = symbol_shorthand.L  # Static and dynamic landmark
X = symbol_shorthand.X  # Camera pose
O = symbol_shorthand.O  # Object Pose
H = symbol_shorthand.H  # Object motion


class Optimizer(object):
    def __init__(self, map: Map):
        self.map = map
        self.poses_set = None
        self.object_set = None
        self.landmark_set = None
        self.dynamic_landmark_set = None

        # GTSAM model noises
        self.model_meas_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.model_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        )
        self.model_point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        self.model_motion_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        )

        # Initial solution noise
        cam_apply_noise = self.map.config["cam"]["noise"]["apply"]
        if cam_apply_noise:
            self.cam_obs_noise = self.map.config["cam"]["noise"]["obs_noise"]
            self.cam_pose_noise = self.map.config["cam"]["noise"]["pose_noise"]
            self.cam_point_noise = self.map.config["cam"]["noise"]["point_noise"]
        else:
            self.cam_obs_noise = 0
            self.cam_pose_noise = 0
            self.cam_point_noise = 0
        obj_apply_noise = self.map.config["car"]["noise"]["apply"]
        if obj_apply_noise:
            self.obj_pose_noise = self.map.config["car"]["noise"]["pose_noise"]
            self.obj_point_noise = self.map.config["car"]["noise"]["point_noise"]
            self.obj_motion_noise = self.map.config["car"]["noise"]["motion_noise"]
        else:
            self.obj_obs_noise = 0
            self.obj_pose_noise = 0
            self.obj_point_noise = 0
            self.obj_motion_noise = 0

        self.graph = None
        self.initial_estimate = None
        self.ground_truth = None
        self.huber_loss = gtsam.noiseModel.mEstimator.Huber(2.432)

    def setup_camera(self):
        self.graph = NonlinearFactorGraph()
        K = Cal3_S2(
            fx=self.map.K[0, 0],
            fy=self.map.K[1, 1],
            s=0,
            u0=self.map.K[0, 2],
            v0=self.map.K[1, 2],
        )

        # Include poses and landmarks
        self.poses_set = set()
        self.constant_pose_set = set()
        self.static_pose_landmark_dict = dict()
        self.static_landmark_set = set()
        self.dynamic_landmark_dict = dict()

        # 1. Add measurements
        # First construct the covisibility graph for each pose
        for j, point in enumerate(self.map.points):
            if len(point.obs) > 2:
                for i in point.obs:
                    if i in self.static_pose_landmark_dict:
                        self.static_pose_landmark_dict[i].append(j)
                    else:
                        self.static_pose_landmark_dict[i] = [j]

        # Then traverse across the graph and make sure that they
        # a min covscore, otherwise add them but as prior factor
        for pose_id in self.static_pose_landmark_dict:
            for point_id in self.static_pose_landmark_dict[pose_id]:
                meas = self.map.points[point_id].obs[pose_id]
                factor = gtsam.CustomFactor(
                    self.model_meas_noise,
                    [X(pose_id), L(point_id)],
                    partial(error_reprojection, [meas, K]),
                )
                self.graph.push_back(factor)
                self.static_landmark_set.add(point_id)
                if len(self.static_pose_landmark_dict[pose_id]) < 3:
                    self.constant_pose_set.add(pose_id)
                else:
                    self.poses_set.add(pose_id)

        # Dynamic points
        if self.map.config["use_dynamic_points"]:
            # TODO: Does it need a check of how many measurements
            # Reprojection error
            for j, point in enumerate(self.map.car.points):
                if len(point.obs) > 2:
                    for i in point.obs:
                        meas = point.obs[i]
                        factor = gtsam.CustomFactor(
                            self.model_meas_noise,
                            [X(i), L(1000 * (i + 1) + j)],
                            partial(error_reprojection, [meas, K]),
                        )
                        self.graph.push_back(factor)
                        if i in self.dynamic_landmark_dict:
                            self.dynamic_landmark_dict[i].append(j)
                        else:
                            self.dynamic_landmark_dict[i] = [j]

            # Object factors
            for pose_id in self.dynamic_landmark_dict:
                # 1. Object motion factor
                # Makes sure there is motion
                if not ((pose_id + 1) in self.dynamic_landmark_dict.keys()):
                    continue
                for point_id in self.dynamic_landmark_dict[pose_id]:
                    # Makes sure the same landmark is seen in two consecutive poses
                    if (point_id) in self.dynamic_landmark_dict[pose_id + 1]:
                        factor = gtsam.CustomFactor(
                            self.model_point_noise,
                            [
                                H(pose_id),
                                L(1000 * (pose_id + 1) + point_id),
                                L(1000 * (pose_id + 2) + point_id),
                            ],
                            partial(error_object_motion),
                        )
                        self.graph.push_back(factor)

                # 2. Object smoother motion
                if (pose_id + 2) in self.dynamic_landmark_dict.keys():
                    factor = gtsam.BetweenFactorPose3(
                        H(pose_id),
                        H(pose_id + 1),
                        Pose3.Identity(),
                        self.model_motion_noise,
                    )

                    self.graph.push_back(factor)

        # 2. Adding priors
        # Add Prior factor
        first_pose_id = next(iter(self.poses_set))
        prior_pose = PriorFactorPose3(
            X(first_pose_id),
            Pose3(self.map.cam_poses[first_pose_id]),
            self.model_pose_noise,
        )
        self.graph.push_back(prior_pose)

        for pose_id in self.constant_pose_set:
            prior_pose = PriorFactorPose3(
                X(pose_id),
                Pose3(self.map.cam_poses[pose_id]),
                self.model_pose_noise,
            )
            self.graph.push_back(prior_pose)

        # Set prior to avoid gauge freedom
        first_point_id = next(iter(self.static_landmark_set))
        prior_landmark = PriorFactorPoint3(
            L(first_point_id),
            Point3(self.map.pts[first_point_id]),
            self.model_point_noise,
        )
        self.graph.push_back(prior_landmark)
        # self.graph.print("Factor Graph:\n")

        # 3. Store initial solution
        self.initial_estimate = Values()
        self.ground_truth = Values()
        rng = np.random.default_rng()

        for pose_id in self.poses_set:
            pose = self.map.cam_poses[pose_id]
            transformed_pose = pose.retract(
                self.cam_pose_noise * rng.standard_normal(6).reshape(6, 1)
            )
            self.initial_estimate.insert(X(pose_id), transformed_pose)
            self.ground_truth.insert(X(pose_id), pose)

        for pose_id in self.constant_pose_set:
            pose = self.map.cam_poses[pose_id]
            transformed_pose = pose.retract(
                self.cam_pose_noise * rng.standard_normal(6).reshape(6, 1)
            )
            self.initial_estimate.insert(X(pose_id), transformed_pose)
            self.ground_truth.insert(X(pose_id), pose)

        for point_id in self.static_landmark_set:
            transformed_point = self.map.pts[
                point_id
            ] + self.cam_point_noise * rng.standard_normal(3)
            self.initial_estimate.insert(L(point_id), Point3(transformed_point))
            self.ground_truth.insert(L(point_id), Point3(self.map.pts[point_id]))

        if self.map.config["use_dynamic_points"]:
            for pose_id in self.dynamic_landmark_dict:
                for point_id in self.dynamic_landmark_dict[pose_id]:
                    pts = self.map.car.points[point_id].hist_pts[pose_id]
                    self.ground_truth.insert(
                        L(1000 * (pose_id + 1) + point_id), Point3(pts)
                    )
                    pts += self.obj_point_noise * rng.standard_normal(3)
                    self.initial_estimate.insert(
                        L(1000 * (pose_id + 1) + point_id), Point3(pts)
                    )
                # Motion initiallation
                if (pose_id + 1) in self.dynamic_landmark_dict.keys():
                    pose1 = self.map.car.car_poses[pose_id]
                    pose2 = self.map.car.car_poses[pose_id + 1]
                    motion = pose2 * pose1.inverse()
                    transformed_motion = motion.retract(
                        self.obj_motion_noise * rng.standard_normal(6).reshape(6, 1)
                    )
                    self.initial_estimate.insert(H(pose_id), transformed_motion)
                    self.ground_truth.insert(H(pose_id), motion)

    def run(self):
        self.setup_camera()

        plotting.plot_hessian_matrix(self.graph)
        # plotting.plot_graph_connectivity(self.graph, self.initial_estimate)

        # Optimize the graph and print results
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("TERMINATION")
        params.setMaxIterations(100)
        optimizer = LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, params
        )
        print("Optimizing:")
        result = optimizer.optimize()
        # result.print("Final results:\n")
        print("initial error = {}".format(self.graph.error(self.initial_estimate)))
        print("final error = {}".format(self.graph.error(result)))

        # Results
        marginals = Marginals(self.graph, result)
        plotting.plot_trajectory_camera(1, result, marginals=marginals, scale=5)
        if self.map.config["use_dynamic_points"]:
            plotting.plot_trajectory_car_per_motion(
                1, self.map.car.car_poses[0], result, marginals=marginals, scale=2
            )
        # plot.plot_3d_points(1, result, marginals=marginals)
        # plotting.plot_3d_points_car(1, self.map.car.pts)

        plotting.plot_results_vs_gt(
            result, self.ground_truth, title="Cam states vs GT."
        )

        plot.set_axes_equal(1)
        plt.show()


if __name__ == "__main__":
    import yaml

    with open("params/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    simulation = Map(config)
    optimizer = Optimizer(simulation)
    optimizer.run()
