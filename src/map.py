import gtsam
import numpy as np
import matplotlib.pyplot as plt


from utils import plotPath, posesOnCircle
from car import Car


class MapPoint(object):
    def __init__(self, id: int, pts: np.ndarray):
        self.id = id
        self.pts = pts  # Assumed to be w.r.t world frame
        self.obs = dict()

    def addObservation(self, pose_id: int, feature: np.ndarray):
        self.obs[pose_id] = feature


class Map(object):
    def __init__(self, config: dict):

        self.K = np.array(config["cam"]["calibration"])
        self.img_width = config["cam"]["image_dims"]["width"]
        self.img_height = config["cam"]["image_dims"]["heigh"]

        self.n_frames = config["cam"]["simulation"]["n_frames"]
        self.n_points = config["cam"]["simulation"]["n_points"]
        self.n_obs = config["cam"]["simulation"]["n_obs"]

        self.apply_noise = config["cam"]["noise"]["apply"]
        if self.apply_noise:
            self.obs_noise = config["cam"]["noise"]["obs_noise"]
            self.pose_noise = config["cam"]["noise"]["pose_noise"]
            self.point_noise = config["cam"]["noise"]["point_noise"]

        self.pts = None
        self.points = None
        self.cam_poses = None
        self.car = Car(config)

        self.T_bc = np.array(config["cam"]["T_bc"])
        self.config = config

        self.simulate()

    def generateTraj(self):
        self.cam_poses = posesOnCircle(self.n_frames, 45)

    def generatePoints(self):
        self.pts = np.random.uniform([-10, -10, -10], [10, 10, 10], (self.n_points, 3))
        self.points = [MapPoint(i, pt) for i, pt in enumerate(self.pts)]

    def generateObs(self):
        # Static observations
        for pose_id in range(self.n_frames):
            projs, point_ids = self.projectPoint2Cam(pose_id)
            for obs, point_id in zip(projs, point_ids):
                self.points[point_id].addObservation(pose_id, obs)

        # Dynamic observations (it assumes that each timestamps had seen
        # the vehicle and its corresponding points w.r.t world frame)
        for pose_id in range(self.n_frames):
            dyn_projs, dyn_point_ids = self.projectCarPoint2Cam(pose_id)
            for obs, point_id in zip(dyn_projs, dyn_point_ids):
                self.car.points[point_id].addObservation(pose_id, obs)

    def projectPoint2Cam(self, pose_id: int):
        """
        Assuming that points are w.r.t world frame (w), transforms them to camera frame
        and them project them to image plane by pinhole camera model
        """
        T_wc = self.cam_poses[pose_id].matrix()
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3].reshape(1, 3)

        R_cw = R_wc.T
        points_cam = (R_cw @ (self.pts - t_wc).T).T
        points_ids = np.arange(0, self.n_points)

        # Keep only points in front of the camera and project
        visible_mask = points_cam[:, 2] > 0.1
        visible_points = points_cam[visible_mask]
        points_ids = points_ids[visible_mask]

        # Project by pinhole model and keep points in camera-view
        projs = self.applyPinholeProj(visible_points)
        projs_mask = (
            (projs[:, 0] > 0)
            & (projs[:, 0] < self.img_width)
            & (projs[:, 1] > 0)
            & (projs[:, 1] < self.img_height)
        )

        return (
            projs[projs_mask],
            points_ids[projs_mask],
        )  # filtered points and correspondence id

    def projectCarPoint2Cam(self, pose_id: int):
        """
        Assuming that points are w.r.t world frame (w), transforms them to camera frame
        and them project them to image plane by pinhole camera model
        """
        T_wc = self.cam_poses[pose_id].matrix()
        R_wc = T_wc[:3, :3]
        t_wc = T_wc[:3, 3].reshape(1, 3)
        R_cw = R_wc.T

        points_cam = np.zeros((len(self.car.points), 3))
        for i, point in enumerate(self.car.points):
            pts = np.expand_dims(point.hist_pts[pose_id], 0)
            pts_cam = (R_cw @ (pts - t_wc).T).T
            points_cam[i] = pts_cam
        points_ids = np.arange(0, len(self.car.points))

        # Keep only points in front of the camera and project
        visible_mask = points_cam[:, 2] > 0.1
        visible_points = points_cam[visible_mask]
        points_ids = points_ids[visible_mask]

        # Project by pinhole model and keep points in camera-view
        projs = self.applyPinholeProj(visible_points)
        projs_mask = (
            (projs[:, 0] > 0)
            & (projs[:, 0] < self.img_width)
            & (projs[:, 1] > 0)
            & (projs[:, 1] < self.img_height)
        )

        return (
            projs[projs_mask],
            points_ids[projs_mask],
        )  # filtered points and correspondence id

    def applyPinholeProj(self, points_3d: np.ndarray) -> np.ndarray:
        projected = points_3d / points_3d[:, 2, np.newaxis]  # Normalize by z
        return (self.K @ projected.T).T[:, :2]

    def plotHessian(self):
        # The Hessian must have a diagonal meaning that
        # the variable connects to itself
        H = np.eye(self.n_frames + self.n_points + self.car.n_points)
        for point in self.points:
            for obs_id in point.obs:
                H[self.n_frames + point.id, obs_id] = 1
        for point in self.car.points:
            for obs_id in point.obs:
                H[self.n_frames + self.n_points + point.id, obs_id] = 1
        H = H + H.T  # Make symmetric

        plt.figure(figsize=(8, 8))
        plt.spy(H, markersize=1)
        plt.title("Sparsity pattern of Hessian (Bundle Adjustment)")
        plt.xlabel("Variables")
        plt.ylabel("Variables")
        plt.show()

    def simulate(self):
        self.generateTraj()
        self.generatePoints()
        self.generateObs()
