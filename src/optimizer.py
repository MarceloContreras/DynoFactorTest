import numpy as np
import gtsam
import matplotlib.pyplot as plt


from map import Map
from utils import plot_trajectory_car, plot_3d_points_car
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

        # GTSAM noises
        self.model_meas_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)
        self.model_pose_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1])
        )
        self.model_point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        # Initial solution noise
        apply_noise = self.map.config["cam"]["noise"]["apply"]
        if apply_noise:
            self.obs_noise = self.map.config["cam"]["noise"]["obs_noise"]
            self.pose_noise = self.map.config["cam"]["noise"]["pose_noise"]
            self.point_noise = self.map.config["cam"]["noise"]["point_noise"]

        self.graph = None
        self.initial_estimate = None
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
                if (len(self.static_pose_landmark_dict[pose_id])<3):
                    self.constant_pose_set.add(pose_id)
                else:
                    self.poses_set.add(pose_id)

        # Dynamic points
        if self.map.config["use_dynamic_points"]:
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
                        self.poses_set.add(i)
                        if i in self.dynamic_landmark_dict:
                            self.dynamic_landmark_dict[i].append(j)
                        else:
                            self.dynamic_landmark_dict[i] = [j]

            # Object factors
            for pose_id in self.dynamic_landmark_dict:
                # Object smoother motion
                if (pose_id + 1) in self.dynamic_landmark_dict.keys():
                    for point_id in self.dynamic_landmark_dict[pose_id]:
                        if (point_id + 1) in self.dynamic_landmark_dict[pose_id]:
                            factor = gtsam.CustomFactor(
                                self.model_point_noise,
                                [
                                    H(pose_id),
                                    L(1000 * (pose_id + 1) + point_id),
                                    L(1000 * (pose_id + 1) + point_id + 1),
                                ],
                                partial(error_object_motion),
                            )
                            self.graph.push_back(factor)

                    # Object motion
                    if (pose_id + 2) in self.dynamic_landmark_dict.keys():
                        print(
                            f"Making factor between {pose_id},{pose_id+1} and {pose_id+1},{pose_id+2}"
                        )
                        factor = gtsam.BetweenFactorPose3(
                            H(pose_id),
                            H(pose_id + 1),
                            Pose3.Identity(),
                            self.model_pose_noise,
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
        self.graph.print("Factor Graph:\n")

        # 3. Store initial solution
        self.initial_estimate = Values()
        rng = np.random.default_rng()

        for pose_id in self.poses_set:
            pose = self.map.cam_poses[pose_id]
            transformed_pose = pose.retract(
                self.pose_noise * rng.standard_normal(6).reshape(6, 1)
            )
            self.initial_estimate.insert(X(pose_id), transformed_pose)

        for pose_id in self.constant_pose_set:
            pose = self.map.cam_poses[pose_id]
            transformed_pose = pose.retract(
                self.pose_noise * rng.standard_normal(6).reshape(6, 1)
            )
            self.initial_estimate.insert(X(pose_id), transformed_pose)

        for point_id in self.static_landmark_set:
            transformed_point = self.map.pts[
                point_id
            ] + self.point_noise * rng.standard_normal(3)
            self.initial_estimate.insert(L(point_id), Point3(transformed_point))

        if self.map.config["use_dynamic_points"]:
            for pose_id in self.dynamic_landmark_dict:
                # TODO: Should I verify if the factor actually exist?
                for point_id in self.dynamic_landmark_dict[pose_id]:
                    pts = self.map.car.points[point_id].hist_pts[pose_id]
                    pts += self.point_noise * rng.standard_normal(3)
                    self.initial_estimate.insert(
                        L(1000 * (pose_id + 1) + point_id), Point3(pts)
                    )
                # Motion initiallation
                if (pose_id + 1) in self.dynamic_landmark_dict.keys():
                    pose1 = self.map.car.car_poses[pose_id]
                    pose2 = self.map.car.car_poses[pose_id + 1]
                    motion = pose2 * pose1.inverse()
                    transformed_motion = motion.retract(
                        self.pose_noise * rng.standard_normal(6).reshape(6, 1)
                    )
                    self.initial_estimate.insert(H(pose_id), transformed_motion)

        self.initial_estimate.print("Initial Estimates:\n")

    def run(self):
        self.setup_camera()

        self.plot_hessian_matrix()

        # Optimize the graph and print results
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("TERMINATION")
        params.setMaxIterations(100)
        optimizer = LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, params
        )
        print("Optimizing:")
        result = (
            optimizer.optimize()
        )  # TODO FIX ME: CheiralityException after including dynamic points and vehicle motion
        result.print("Final results:\n")
        print("initial error = {}".format(self.graph.error(self.initial_estimate)))
        print("final error = {}".format(self.graph.error(result)))

        marginals = Marginals(self.graph, result)
        # plot.plot_3d_points(1, result, marginals=marginals)
        plot.plot_trajectory(1, result, marginals=marginals, scale=5)
        plot_trajectory_car(1, self.map.car.car_poses, scale=2)
        plot_3d_points_car(1, self.map.car.pts)
        plot.set_axes_equal(1)
        plt.show()

    def plot_hessian_matrix(self):
        keys = set()
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            for k in factor.keys():
                keys.add(k)
        keys = sorted(keys)

        # Build mapping key -> index
        key_to_idx = {k: i for i, k in enumerate(keys)}

        # Initialize adjacency matrix
        A = np.ones((len(keys), len(keys)), dtype=int)

        # Fill adjacency by checking which variables appear together in a factor
        for i in range(self.graph.size()):
            factor = self.graph.at(i)
            f_keys = list(factor.keys())
            for a in range(len(f_keys)):
                for b in range(a + 1, len(f_keys)):
                    ia = key_to_idx[f_keys[a]]
                    ib = key_to_idx[f_keys[b]]
                    A[ia, ib] = 0
                    A[ib, ia] = 0
           
        np.fill_diagonal(A, 0)

        plt.figure(figsize=(4,4))
        plt.imshow(A, cmap='gray', interpolation='none')
        plt.xticks(range(len(keys)), [gtsam.DefaultKeyFormatter(k) for k in keys], rotation=45)
        plt.yticks(range(len(keys)), [gtsam.DefaultKeyFormatter(k) for k in keys])
        plt.title("Variable Connectivity Matrix (Reduced)")
        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    import yaml

    with open("params/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    simulation = Map(config)
    optimizer = Optimizer(simulation)
    optimizer.run()
