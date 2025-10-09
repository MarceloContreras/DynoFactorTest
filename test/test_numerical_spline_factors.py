# pylint: disable=invalid-name, no-name-in-module

import numpy as np

from gtsam import Pose3, Rot3, Point3, Cal3_S2, PinholePoseCal3_S2
from gtsam.utils.numerical_derivative import numericalDerivative21,numericalDerivative22

def test_reproj_err_jacobian():
    def h(camera:PinholePoseCal3_S2, point: Point3):
        return camera.project(point)

    # Values from testPose3.cpp
    R = Rot3.Rodrigues(0.3,0,0) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t)  

    # Values for projection
    K = Cal3_S2(1.0,1.0,0.0,0.0,0.0)
    cam = PinholePoseCal3_S2(T, K)

    # Landmark
    P = Point3(-1.0,0.7,7.5) 
    P_w = np.transpose(np.array([[-1.0,0.7,7.5,1.0]]))

    # Analytic jacobian
    P_c = np.squeeze(T.inverse().matrix()@P_w)
    x,y,z = P_c[0],P_c[1],P_c[2]
    
    dpi_dPc = 1/z * np.array([[1,0,-x/z],
                              [0,1,-y/z]])
    dPc_dT = np.array([[ 0,-z, y,-1, 0, 0],
                       [ z, 0,-x, 0,-1, 0],
                       [-y, x, 0, 0, 0,-1]])   
    analytic_H1 = dpi_dPc@dPc_dT

    # Numerical jacobian
    numerical_H1 = numericalDerivative21(h, cam, P)
    
    print(np.allclose(numerical_H1,analytic_H1))


def test_object_motion_jacobian():
    def h(obj_pose:Pose3, omega:np.ndarray):
        adj = obj_pose.AdjointMap()
        T = Pose3.Expmap(adj@omega)
        point = Point3(-1.0,0.7,7.5) 
        return T.transformFrom(point)

    # Values from testPose3.cpp
    R = Rot3.Rodrigues(0.3,0,0) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t)  

    # Point
    point_homo = np.transpose(np.array([-1.0,0.7,7.5,1]))

    # Velocity
    w = np.transpose(np.array([[0.1,0.1,0.1,-0.1,-0.1,1.0]]))

    ## Analytic jacobian
    AdjT_w = T.AdjointMap()@w

    # Jacobian of action w.r.t 
    T_k = Pose3.Expmap(AdjT_w).matrix()
    R_k = T_k[:3,:3]
    skew_P_c = np.array([[      0, -point_homo[2],  point_homo[1]],
                         [ point_homo[2],       0, -point_homo[0]],
                         [-point_homo[1],  point_homo[0],       0]])
    dPc_dT = np.hstack((-R_k@skew_P_c, R_k))  

    # Jacobian of Exp(e) w.r.t e
    dExpMap_de = Pose3.ExpmapDerivative(AdjT_w)

    # Jacobian of Adj w.r.t T
    dAdj_dT = -T.AdjointMap()@Pose3.adjointMap(w)

    # Rule of chain
    analytic_H1 = dPc_dT@dExpMap_de@dAdj_dT

    # Numerical Jacobian 
    numerical_H1 = numericalDerivative21(h, T, w)

    print(np.allclose(numerical_H1,analytic_H1)) 

def test_constraint_factor():
    Gx = np.array([[0,0,0,1,0,0]])
    Gtheta = np.array([[0,0,1,0,0,0]])

    def h(pose:Pose3, T_dot:np.ndarray):
        tangent = Pose3.Logmap(pose)
        vec = np.expand_dims(Pose3.Vee(T_dot),axis=-1) # Problem since this is not a lie algebra        
        return np.expand_dims(np.cos(Gtheta@tangent),axis=-1)@Gx@vec

    # Pose
    R = Rot3.Rodrigues(0.3,0.3,0.3) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t) 

    # T_dot = omega_hat @ T
    T_dot = np.array([[   0,-0.1,   0,-0.1],
                      [ 0.1,   0,-0.1,-0.1],
                      [-0.1, 0.1,   0, 1.0],
                      [   0,   0,   0, 1.0]])
    T_dot = T_dot@T.matrix()
    vec = np.expand_dims(Pose3.Vee(T_dot),axis=-1)
    
    # Analytic jacobian
    dLog_dt = Pose3.LogmapDerivative(T)
    analytic_H1 = -Gx@vec@np.expand_dims(np.sin(Gtheta@Pose3.Logmap(T)),axis=-1)@Gtheta@dLog_dt    

    # Numerical Jacobian 
    numerical_H1 = numericalDerivative21(h, T, T_dot)
 
    print(np.allclose(numerical_H1,analytic_H1)) 

if __name__ == "__main__":
    test_reproj_err_jacobian()
    test_object_motion_jacobian()
    test_constraint_factor()