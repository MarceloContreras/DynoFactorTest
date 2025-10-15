import numpy as np
import gtsam
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
from factors import error_reprojection

L = symbol_shorthand.L
X = symbol_shorthand.X


class Optimizer(object):
    def __init__(self, map: Map):
        self.map = map
        self.poses_set = None
        self.object_set = None
        self.landmark_set = None
        self.dynamic_landmark_set = None

        # GTSAM noises
        self.model_meas_noise = gtsam.noiseModel.Isotropic.Sigma(
            2, 1.0
        )  # one pixel in u and v
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
        # robust_model = gtsam.noiseModel.Robust.Create(self.huber_loss,self.model_meas_noise)

        # Include poses and landmarks
        self.poses_set = set()
        self.landmark_set = set()

        for j, point in enumerate(self.map.points):
            if len(point.obs) > 2:
                for pose_id in point.obs:
                    meas = point.obs[pose_id]
                    factor = gtsam.CustomFactor(
                        self.model_meas_noise,
                        [X(pose_id), L(j)],
                        partial(error_reprojection, [meas, K]),
                    )
                    self.graph.push_back(factor)
                    self.poses_set.add(pose_id)
                self.landmark_set.add(j)

        # Add Prior factor
        first_pose_id = next(iter(self.poses_set))
        prior_pose = PriorFactorPose3(
            X(first_pose_id),
            Pose3(self.map.cam_poses[first_pose_id]),
            self.model_pose_noise,
        )
        self.graph.push_back(prior_pose)

        # Set prior to avoid gauge freedom
        first_point_id = next(iter(self.landmark_set))
        prior_landmark = PriorFactorPoint3(
            L(first_point_id),
            Point3(self.map.pts[first_point_id]),
            self.model_point_noise,
        )
        self.graph.push_back(prior_landmark)
        self.graph.print("Factor Graph:\n")

        # Store initial solution
        self.initial_estimate = Values()
        rng = np.random.default_rng()
        for pose_id in self.poses_set:
            pose = self.map.cam_poses[pose_id]
            transformed_pose = pose.retract(
                self.pose_noise * rng.standard_normal(6).reshape(6, 1)
            )
            self.initial_estimate.insert(X(pose_id), transformed_pose)
            # self.map.cam_poses[pose_id] = transformed_pose.matrix()

        for point_id in self.landmark_set:
            transformed_point = self.map.pts[
                point_id
            ] + self.point_noise * rng.standard_normal(3)
            self.initial_estimate.insert(L(point_id), Point3(transformed_point))
            # self.map.pts[point_id] = transformed_point

        self.initial_estimate.print("Initial Estimates:\n")

    def run(self):
        self.setup_camera()

        # Optimize the graph and print results
        params = gtsam.LevenbergMarquardtParams()
        params.setVerbosity("TERMINATION")
        optimizer = LevenbergMarquardtOptimizer(
            self.graph, self.initial_estimate, params
        )
        print("Optimizing:")
        result = optimizer.optimize()
        result.print("Final results:\n")
        print("initial error = {}".format(self.graph.error(self.initial_estimate)))
        print("final error = {}".format(self.graph.error(result)))

        marginals = Marginals(self.graph, result)
        plot.plot_3d_points(1, result, marginals=marginals)
        plot.plot_trajectory(1, result, marginals=marginals, scale=8)
        plot.set_axes_equal(1)
        plt.show()


if __name__ == "__main__":
    import yaml

    with open("params/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    test = Map(config)
    test.simulate()
    optimizer = Optimizer(test)
    optimizer.run()

# TODO: Check FrontEnd.cc class in DynoSam
