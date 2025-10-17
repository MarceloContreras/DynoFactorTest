import gtsam
import numpy as np

from gtsam import Point3, Pose3, Rot3


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
        self.x, self.y, self.theta = -10, -10, np.pi / 2

        self.T_bc = np.array(config["car"]["T_bc"])

        self.apply_noise = config["car"]["noise"]["apply"]
        if self.apply_noise:
            self.pose_noise = config["car"]["noise"]["pose_noise"]
            self.point_noise = config["car"]["noise"]["point_noise"]

        self.simulate()

    def generateTraj(self):
        init_rotation = Rot3.Ypr(np.pi / 2, 0, 0)
        init_position = np.array([-10, -10, 0])
        init = Pose3(init_rotation, init_position)

        poses = [init]
        for _ in range(1, self.n_poses):
            # Delta step
            dx = self.v * self.dt
            dtheta = self.v / self.L * np.tan(self.delta) * self.dt

            # Delta rotation in world frame
            delta_rotation = Rot3.Ypr(dtheta, 0, 0)
            # Delta translation in local frame
            delta_translation_local = np.array([dx, 0, 0])
            # Define delta pose
            delta_pose = Pose3(delta_rotation, delta_translation_local)

            # Compose pose
            poses.append(poses[-1].compose(delta_pose))

        self.car_poses = poses

    def generatePoints(self):
        local_pts = np.random.uniform([-5, -5, -5], [5, 5, 5], (self.n_points, 3))
        self.points = [DynamicMapPoint(i) for i in range(self.n_points)]

        for i in range(self.n_poses):
            car_pose = self.car_poses[i].matrix()
            car_rot = car_pose[:3, :3]
            car_pos = car_pose[:3, 3]
            world_pts = (car_rot @ local_pts.T).T + car_pos

            for j, mp in enumerate(self.points):
                mp.addPoseHistory(world_pts[j])

            # Stacks all the points across different timestamps (overlap)
            self.pts.append(world_pts)

        self.pts = np.stack(self.pts, axis=0)

    def simulate(self):
        self.generateTraj()
        self.generatePoints()

    def reset(self):
        self.x = 0
        self.y = 0
        self.theta = 0
