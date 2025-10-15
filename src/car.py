import gtsam
import numpy as np


class DynamicMapPoint(object):
    def __init__(self, id: int):
        self.id = id
        self.hist_pts = []  # Assumed to be w.r.t world frame
        self.obs = dict()

    def addPoseHistory(self, pose: np.ndarray):
        self.hist_pts.append(pose)

    def addObservation(self, pose_id: int, feature: np.ndarray):
        self.obs[pose_id] = feature


class Car(object):
    def __init__(self, config: dict):

        self.L = config["car"]["params"]["l"]
        self.v = config["car"]["params"]["v"]
        self.dt = config["car"]["params"]["dt"]
        self.delta = config["car"]["params"]["delta"]

        self.n_points = config["car"]["simulation"]["n_points"]
        self.n_poses = config["car"]["simulation"]["n_frames"]

        self.pts = []
        self.points = []
        self.car_poses = []
        self.x, self.y, self.theta = 0, 0, 0.785398

        self.T_bc = np.array(config["car"]["T_bc"])

        self.apply_noise = config["car"]["noise"]["apply"]
        if self.apply_noise:
            self.pose_noise = config["car"]["noise"]["pose_noise"]
            self.point_noise = config["car"]["noise"]["point_noise"]

    def generateTraj(self):
        for _ in range(self.n_poses):
            T = np.array(
                [
                    [np.cos(self.theta), -np.sin(self.theta), 0, self.x],
                    [np.sin(self.theta), np.cos(self.theta), 0, self.y],
                    [0, 0, 1, 0.5],
                    [0, 0, 0, 1.0],
                ]
            )

            if self.apply_noise:
                mean = np.zeros(6)
                cov = np.diag([self.pose_noise] * 3 + [self.pose_noise] * 3)
                noise = np.random.multivariate_normal(mean, cov)
                noise[2] = 0  # trans-z
                noise[3] = 0  # rot-x
                noise[4] = 0  # rot-y

                local_perturb = gtsam.Pose3.Expmap(noise)
                pose_gtsam = gtsam.Pose3(T)
                T = local_perturb.compose(
                    pose_gtsam
                ).matrix()  # TODO: Should not be in another way

            self.stepBicycleModel()
            self.car_poses.append(T @ self.T_bc)

        self.car_poses = np.stack(self.car_poses, axis=0)

    def generatePoints(self):
        local_pts = np.random.uniform([-1, -1, -3], [1, 1, 3], (self.n_points, 3))
        if self.apply_noise:
            local_pts += np.random.normal(
                loc=0, scale=self.point_noise, size=local_pts.shape
            )

        self.points = [DynamicMapPoint(i) for i in range(self.n_points)]

        for i in range(self.n_poses):
            car_rot = self.car_poses[i, :3, :3]
            car_pos = self.car_poses[i, :3, 3]
            world_pts = (car_rot @ local_pts.T).T + car_pos

            for j, mp in enumerate(self.points):
                mp.addPoseHistory(world_pts[j])

            # Stacks all the points across different timestamps (overlap)
            self.pts.append(world_pts)

        self.pts = np.stack(self.pts, axis=0)

    def stepBicycleModel(self):
        self.x = self.x + self.v * np.cos(self.theta) * self.dt
        self.y = self.y + self.v * np.sin(self.theta) * self.dt
        self.theta = self.theta + (self.v / self.L) * np.tan(self.delta) * self.dt

    def simulate(self):
        self.generateTraj()
        self.generatePoints()

    def reset(self):
        self.x = 0
        self.y = 0
        self.theta = 0
