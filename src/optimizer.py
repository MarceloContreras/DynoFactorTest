import numpy as np
import gtsam
import matplotlib.pyplot as plt

from map import Map
from gtsam import symbol_shorthand
from gtsam import (Cal3_S2, DoglegOptimizer,
                   NonlinearFactorGraph,
                   PriorFactorPoint3, PriorFactorPose3, Values,
                   Point3,Pose3)
from functools import partial
from factors import error_reprojection,error_object_motion,error_object_smoother

L = symbol_shorthand.L
X = symbol_shorthand.X
O = symbol_shorthand.O

class Optimizer(object):
    def __init__(self,map:Map):
        self.map = map
        self.poses_set = None
        self.object_set = None
        self.landmark_set = None
        self.dynamic_landmark_set = None

        self.measurement_noise = gtsam.noiseModel.Isotropic.Sigma(2, 1.0)  # one pixel in u and v
        self.pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.3, 0.3, 0.3, 0.1, 0.1, 0.1]))
        self.point_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)

        self.graph = None
        self.initial_estimate = None

    def setup_camera(self):
        self.graph = NonlinearFactorGraph()

        # Prior factor
        factor = PriorFactorPose3(X(0), Pose3(self.map.cam_poses[0]), self.pose_noise)
        self.graph.push_back(factor)

        K = Cal3_S2(self.map.K[0,0],self.map.K[1,1],0,self.map.K[0,2],self.map.K[1,2])
        
        # Include poses and landmarks
        self.poses_set = set()
        self.landmark_set = set()
        
        for j,point in enumerate(self.map.points):
            for pose_id in point.obs:
                meas = point.obs[pose_id]
                factor = gtsam.CustomFactor(self.measurement_noise,
                                            [X(pose_id),L(j)],
                                            partial(error_reprojection,[meas,K]))
                self.graph.push_back(factor)

                self.poses_set.add(pose_id)
            if len(point.obs):
                self.landmark_set.add(j)

        # Set prior to avoid gauge freedom
        factor = PriorFactorPoint3(L(0), Point3(self.map.pts[0]), self.point_noise)
        self.graph.push_back(factor)
        self.graph.print("Factor Graph:\n")

        # Store initial solution
        self.initial_estimate = Values()
        rng = np.random.default_rng()
        for pose_id in self.poses_set:
            temp = Pose3(self.map.cam_poses[pose_id])
            transformed_pose = temp.retract(0.2 * rng.standard_normal(6).reshape(6, 1))
            self.initial_estimate.insert(X(pose_id), transformed_pose)
            self.map.cam_poses[pose_id] = transformed_pose.matrix()
            
        for point_id in self.landmark_set:
            transformed_point = self.map.pts[point_id] + 0.1 * rng.standard_normal(3)
            self.initial_estimate.insert(L(point_id), Point3(transformed_point))
            self.map.pts[j] = transformed_point
        self.initial_estimate.print("Initial Estimates:\n")

    def setup_objects(self):
        K = Cal3_S2(self.map.K[0,0],self.map.K[1,1],0,self.map.K[0,2],self.map.K[1,2])

        self.dynamic_landmark_set = set()

        # Reprojection error 
        for j,point in enumerate(self.map.car.points):
            for pose_id in point.obs:
                meas = point.obs[pose_id]
                factor = gtsam.CustomFactor(self.measurement_noise,
                                            [X(pose_id),L(j+len(self.map.points))],
                                            partial(error_reprojection,[meas,K]))
                self.graph.push_back(factor)
                self.poses_set.add(pose_id)
            self.dynamic_landmark_set.add(j+len(self.map.points))

        # Object motion error
        for j in range(len(self.map.car.points)-1):
            for i in range(len(self.map.car.car_poses)-1):
                factor = gtsam.CustomFactor(self.measurement_noise,
                                            [O(i),
                                             O(i+1),
                                             L(j+len(self.map.points)),
                                             L(j+len(self.map.points)+1)],
                                            error_object_motion)
                self.graph.push_back(factor)
                self.object_set.add(i)
                self.object_set.add(i+1)

            self.dynamic_landmark_set.add(j+len(self.map.points))
            self.dynamic_landmark_set.add(j+len(self.map.points)+1)

        # Object motion smoother
        for i in range(len(self.map.car.car_poses)-2):
            factor = gtsam.CustomFactor(self.measurement_noise,
                                        [O(i),
                                         O(i+1),
                                         O(i+2)],
                                        error_object_smoother)
            self.graph.push_back(factor)
            self.object_set.add(i)
            self.object_set.add(i+1)
            self.object_set.add(i+2)

        # Initial solution 
        rng = np.random.default_rng()
        for pose_id in self.object_set:
            temp = Pose3(self.map.car.car_poses[pose_id])
            transformed_pose = temp.retract(0.2 * rng.standard_normal(6).reshape(6, 1))
            self.initial_estimate.insert(O(pose_id), transformed_pose)
            self.map.car.car_poses[pose_id] = transformed_pose.matrix()
            
        for point_id in self.dynamic_landmark_set:
            transformed_point = self.map.pts[point_id] + 0.1 * rng.standard_normal(3)
            self.initial_estimate.insert(L(point_id), Point3(transformed_point))
            self.map.pts[j] = transformed_point
        self.initial_estimate.print("Initial Estimates:\n")

    def run(self):
        self.setup_camera()
        self.setup_objects()

        self.map.viz()

         # Optimize the graph and print results
        params = gtsam.DoglegParams()
        params.setVerbosity("TERMINATION")
        optimizer = DoglegOptimizer(self.graph, self.initial_estimate, params)
        print("Optimizing:")
        result = optimizer.optimize()
        result.print("Final results:\n")
        print("initial error = {}".format(self.graph.error(self.initial_estimate)))
        print("final error = {}".format(self.graph.error(result)))

        for i in self.poses_set:
            self.map.cam_poses[i] = result.atPose3(X(i)).matrix()
        for j in self.landmark_set:
            self.map.pts[j] = np.array(result.atPoint3(L(j)))

        self.map.viz()

if __name__ == "__main__":
    import yaml

    with open("params/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    test = Map(config)   
    test.simulate()
    optimizer = Optimizer(test)
    optimizer.run()    


"""
Check FrontEnd.cc class in DynoSam
"""