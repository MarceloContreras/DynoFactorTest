import gtsam
import numpy as np
import matplotlib.pyplot as plt


from utils import plotPath
from car import Car

class MapPoint(object):
    def __init__(self,id:int,pts:np.ndarray):
        self.id = id
        self.pts = pts # Assumed to be w.r.t world frame
        self.obs = dict()
    
    def addObservation(self,pose_id:int,feature:np.ndarray):
        self.obs[pose_id] = feature

class Map(object):
    def __init__(self,config:dict):

        self.K = np.array(config["cam"]["calibration"])
        self.img_width = config["cam"]["image_dims"]["width"]
        self.img_height = config["cam"]["image_dims"]["heigh"]

        self.n_frames = config["cam"]["simulation"]["n_frames"]
        self.n_points = config["cam"]["simulation"]["n_points"]
        self.n_obs = config["cam"]["simulation"]["n_obs"]
        self.t_values = np.linspace(0, 2 * np.pi, self.n_frames)

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

    def generateTraj(self):
        t_stack = np.stack([ # Circle path
            7 * np.cos(self.t_values + np.pi/4),
            7 * np.sin(self.t_values + np.pi/4),
            1.5 + np.sin(0.5 * self.t_values)
            ], axis=-1)
        R_stack = [self.rotZ(t + np.pi/4) for t in self.t_values]
        R_stack = np.stack(R_stack, axis=0)

        cam_poses = np.zeros((self.n_frames,4,4))
        cam_poses[:,:3,:3] = R_stack
        cam_poses[:,:3, 3] = t_stack
        cam_poses[:, 3, 3] = 1.0

        # Extrinsic transformation, necessary since the path was formulated in body-frame
        self.cam_poses = [T_b@self.T_bc for T_b in cam_poses] # Check this
        self.cam_poses = np.stack(self.cam_poses, axis=0)

        if self.apply_noise:
            mean = np.zeros(6)
            cov = np.diag([self.pose_noise]*3 + [self.pose_noise]*3)

            for i in range(self.n_frames):
                noise = np.random.multivariate_normal(mean, cov)
                local_perturb = gtsam.Pose3.Expmap(noise)

                pose_gtsam = gtsam.Pose3(self.cam_poses[i])
                perturb_pose = local_perturb.compose(pose_gtsam)
                self.cam_poses[i] = perturb_pose.matrix()


    def generatePoints(self):
        self.pts = np.random.uniform([-30, -30, -10], [30, 30, 20], (self.n_points, 3))
        if self.apply_noise:
            self.pts += np.random.normal(loc=0, scale=self.point_noise, size=self.pts.shape)
        
        self.points = [MapPoint(i,pt) for i,pt in enumerate(self.pts)]

    def generateObs(self):
        for pose_id in range(self.n_frames):
            # Static points
            projs,point_ids = self.projectPoint2Cam(pose_id)
            if self.apply_noise:
                projs += np.random.normal(loc=0, scale=self.obs_noise, size=projs.shape)

            for obs_id in range(self.n_obs):
                if obs_id > len(point_ids) - 1:
                    break
                self.points[point_ids[obs_id]].addObservation(pose_id,projs[obs_id])
            
            # Dynamic points
            dyn_projs,dyn_point_ids = self.projectDynPoint2Cam(pose_id)
            if self.apply_noise:
                dyn_projs += np.random.normal(loc=0, scale=self.obs_noise, size=dyn_projs.shape)

            for obs_id in range(self.n_obs):
                if obs_id > len(dyn_point_ids) - 1:
                    break
                self.car.points[dyn_point_ids[obs_id]].addObservation(pose_id,dyn_projs[obs_id])

    def projectPoint2Cam(self,pose_id:int):
        """
        Assuming that points are w.r.t world frame (w), transforms them to camera frame
        and them project them to image plane by pinhole camera model
        """        
        T_wc = self.cam_poses[pose_id]
        R_wc = T_wc[:3,:3]
        t_wc = T_wc[:3,3].reshape(1,3)

        R_cw = R_wc.T
        points_cam = (R_cw @ (self.pts - t_wc).T).T
        points_ids = np.arange(0,self.n_points)
        
        # Keep only points in front of the camera and project 
        visible_mask = points_cam[:, 2] > 0.1
        visible_points = points_cam[visible_mask]
        points_ids = points_ids[visible_mask]

        # Project by pinhole model and keep points in camera-view
        projs = self.applyPinholeProj(visible_points)
        projs_mask = (projs[:,0] > 0) & (projs[:,0] < self.img_width) & \
                     (projs[:,1] > 0) & (projs[:,1] < self.img_height) 
        
        return projs[projs_mask],points_ids[projs_mask] #filtered points and correspondence id 
    
    def projectDynPoint2Cam(self,pose_id:int):
        T_wc = self.cam_poses[pose_id]
        R_wc = T_wc[:3,:3]
        t_wc = T_wc[:3,3].reshape(1,3)

        R_cw = R_wc.T
        points_cam = (R_cw @ (self.car.pts[pose_id,...] - t_wc).T).T
        points_ids = np.arange(0,self.car.n_points)
        
        # Keep only points in front of the camera and project 
        visible_mask = points_cam[:, 2] > 0.1
        visible_points = points_cam[visible_mask]
        points_ids = points_ids[visible_mask]

        # Project by pinhole model and keep points in camera-view
        projs = self.applyPinholeProj(visible_points)
        projs_mask = (projs[:,0] > 0) & (projs[:,0] < self.img_width) & \
                     (projs[:,1] > 0) & (projs[:,1] < self.img_height) 
        
        return projs[projs_mask],points_ids[projs_mask] #filtered points and correspondence id 


    def applyPinholeProj(self,points_3d:np.ndarray)->np.ndarray:
        projected = points_3d / points_3d[:, 2, np.newaxis]  # Normalize by z
        return (self.K @ projected.T).T[:, :2]        

    def rotZ(self,theta:float)->np.ndarray:
        return np.array([[np.cos(theta),-np.sin(theta),0],
                         [np.sin(theta), np.cos(theta),0],
                         [            0,             0,1]])
    
    def plotHessian(self):
        H = np.eye(self.n_frames + self.n_points + self.car.n_points) # The Hessian must have a diagonal meening that the variable connects to itself
        for point in self.points:
            for obs_id in point.obs:
                H[self.n_frames + point.id,obs_id] = 1
        for point in self.car.points:
            for obs_id in point.obs:
                H[self.n_frames + self.n_points + point.id,obs_id] = 1
        H = H + H.T # Make symmetric

        plt.figure(figsize=(8, 8))
        plt.spy(H, markersize=1)
        plt.title('Sparsity pattern of Hessian (Bundle Adjustment)')
        plt.xlabel('Variables')
        plt.ylabel('Variables')
        plt.show()

    def simulate(self):
        self.car.simulate()
        self.generateTraj()
        self.generatePoints()
        self.generateObs()
        self.plotHessian()

    def viz(self,plot_objects=True):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot trajectory
        plotPath(self.cam_poses[:,:3,3],self.cam_poses[:,:3,:3], ax, "Camera Trajectory", color='black',axis_length=0.5)
        # Plot static 3D points
        ax.scatter(self.pts[:, 0], self.pts[:, 1], self.pts[:, 2], label='Static Landmarks', color='green', s=5, alpha=0.6)

        if plot_objects:
            # Plot Object trajectory
            plotPath(self.car.car_poses[:,:3,3],self.car.car_poses[:,:3,:3], ax, "Car Trajectory", color = "red", axis_length=0.5)
            # Plot Dynaic 3D points
            ax.scatter(self.car.pts[:,:,0], self.car.pts[:,:,1], self.car.pts[:,:,2],
                         label='Car-Attached Landmarks', color='orange', s=5, alpha=0.6)

        ax.set_title('3D Trajectories of Camera')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.grid(True)
        plt.show()

    def viz_incremental(self):
        fig = plt.figure(0)
        if not fig.axes:
            axes = fig.add_subplot(projection='3d')
        else:
            axes = fig.axes[0]
        plt.cla()
        axes.set_xlim3d(-30, 45)
        axes.set_ylim3d(-30, 45)
        axes.set_zlim3d(-30, 45)

        for i in range(self.n_frames):
            plotPath(np.expand_dims(self.cam_poses[i][:3,3],axis=0),
                     np.expand_dims(self.cam_poses[i][:3,:3],axis=0), 
                     axes, color='black',axis_length=0.5,no_legend=True)
            plotPath(np.expand_dims(self.car.car_poses[i][:3,3],axis=0),
                     np.expand_dims(self.car.car_poses[i][:3,:3],axis=0), 
                     axes, color='black',axis_length=0.5,no_legend=True)
            plt.pause(1)


# //TODO: Change height 
    