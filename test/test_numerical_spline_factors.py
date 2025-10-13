# pylint: disable=invalid-name, no-name-in-module

import numpy as np

from gtsam import Pose3, \
    Rot3, Point3, Cal3_S2, PinholePoseCal3_S2
from gtsam.utils.numerical_derivative import numericalDerivative21,\
    numericalDerivative22,numericalDerivative11

def test_reproj_err_jacobian():
    # Section IV.A
    def h(camera:PinholePoseCal3_S2, point: Point3):
        return camera.project(point)

    # Values from testPose3.cpp
    R = Rot3.Rodrigues(0.3,0,0) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t)  

    # Values for projection
    K = Cal3_S2(1.0,1.0,0.0,0.0,0.0)
    cam = PinholePoseCal3_S2(T, K)

    # Landmark w.r.t {w} frame and {c} frame
    P = Point3(-1.0,0.7,7.5) 
    P_w = np.transpose(np.array([[-1.0,0.7,7.5,1.0]]))
    P_c = np.squeeze(T.inverse().matrix()@P_w)
    x,y,z = P_c[0],P_c[1],P_c[2]
    
    # Analytic jacobian
    dpi_dPc = 1/z * np.array([[1,0,-x/z],
                              [0,1,-y/z]])
    dPc_dT = np.array([[ 0,-z, y,-1, 0, 0],
                       [ z, 0,-x, 0,-1, 0],
                       [-y, x, 0, 0, 0,-1]])   
    analytic_H1 = dpi_dPc@dPc_dT

    # Numerical jacobian
    numerical_H1 = numericalDerivative21(h, cam, P)
    
    assert np.allclose(numerical_H1,analytic_H1)


def test_object_motion_jacobian():
    #! This error actually lacks of the differente between point_k and point_k-1
    #! but since we are interest just in H@point_k-1 jacobian, we omit the first point
    # Section IV.B 
    def h(obj_pose:Pose3, omega_bar:np.ndarray):
        adj = obj_pose.AdjointMap()
        H = Pose3.Expmap(adj@omega_bar)
        point = Point3(-1.0,0.7,7.5) 
        return H.transformFrom(point)

    # Values from testPose3.cpp for object pose
    O_Rot = Rot3.Rodrigues(0.3,0,0) 
    O_trans = Point3(3.5,-2.2,4.2)      
    O = Pose3(O_Rot,O_trans)  

    # Point
    point_homo = np.array([-1.0, 0.7, 7.5, 1])
    point_homo = np.transpose(point_homo)

    # Velocity
    omega_bar = np.array([0.1, 0.1, 0.1, -0.1, -0.1, 1.0])
    omega_bar = np.transpose(omega_bar)

    ## Analytic jacobian
    AdjO_omega = O.AdjointMap()@omega_bar

    # 1. Jacobian of action w.r.t H
    H = Pose3.Expmap(AdjO_omega)
    H_Rot = H.rotation().matrix()
    skew_P_c = np.array([[      0, -point_homo[2],  point_homo[1]],
                         [ point_homo[2],       0, -point_homo[0]],
                         [-point_homo[1],  point_homo[0],       0]])
    dPc_dH = np.hstack((-H_Rot@skew_P_c, H_Rot))  

    # 2. Jacobian of Exp(e) w.r.t (e) (simply Left SE(3) jacobian evaluated at AdjT_w)
    dExpMap_de = Pose3.ExpmapDerivative(AdjO_omega)

    # 3. Jacobian of Adj w.r.t O (object pose) and omega_bar (generalized velocity)
    # dAdj_dO = -O.AdjointMap() @ Pose3.adjointMap(omega_bar)
    # dAdj_domega = O.AdjointMap()
    dAdj_dO = np.zeros((6,6), order='F', dtype=float)
    dAdj_domega = np.zeros((6,6), order='F', dtype=float)
    O.Adjoint(omega_bar,dAdj_dO,dAdj_domega)

    # Final result * Rule of chain
    analytic_H1 = dPc_dH@dExpMap_de@dAdj_dO
    analytic_H2 = dPc_dH@dExpMap_de@dAdj_domega

    # Numerical Jacobian 
    numerical_H1 = numericalDerivative21(h, O, omega_bar)
    numerical_H2 = numericalDerivative22(h, O, omega_bar)
    
    assert np.allclose(numerical_H1,analytic_H1) and np.allclose(numerical_H2,analytic_H2)

def test_constraint_factor():
    # Section IV.C 
    Gx = np.array([[0,0,0,0,0,0,0,0,0,1,0,0]])
    Gtheta = np.array([[0,0,1,0,0,0]])
    G  = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0. , 0.,  1.,  0.,  0.,  0.],
                   [ 0. ,-1.,  0.,  0.,  0.,  0.],
                   [ 0. , 0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0., -0.],
                   [ 1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]])
    
    def vec(M:np.ndarray):
        """Vectorize top 3x4 block (12x1), column-major"""
        return M[0:3,:].reshape(-1, order='F')

    def h(pose:Pose3, omega:np.ndarray):
        T_dot = Pose3.Hat(omega) @ pose.matrix()
        T_dot_vec = vec(T_dot)
        angle_z = np.dot(Gtheta,Pose3.Logmap(pose))
        return np.array([np.cos(angle_z)@Gx@T_dot_vec])

    # Pose
    R = Rot3.Rodrigues(0.3,0.3,0.3) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t) 

    # Omega (generalized velocity)
    rho = np.array([10.0, -0.2, 1.5])   
    phi = np.array([0.01, np.pi/8, -0.03]) 
    omega = np.hstack((phi, rho))          

    # Additional variables
    theta = np.dot(Gtheta,Pose3.Logmap(T))
    sin, cos = np.sin(theta),np.cos(theta)
    Tdot = Pose3.Hat(omega) @ T.matrix()
    Tdot_vec = vec(Tdot)
    
    # Analytic Jacobians 
    Jr_inv = Pose3.LogmapDerivative(T)
    dDTvec_dTvec = np.kron(np.eye(4), Pose3.Hat(omega)[:3,:3])
    dTvec_domega = np.kron(np.eye(4), T.rotation().matrix()) @ G

    # 1. Chain rule
    # 1.1 d(cos(theta))/d(T) * Gx * Tdot_vec 
    temp1 = -sin*Gx@Tdot_vec*Gtheta@Jr_inv
    # 1.2 cos(theta) * Gx * d(Tdot_vec)/d(T)
    temp2 = cos@Gx@dDTvec_dTvec@dTvec_domega

    analytic_H1 = temp1 + temp2
    
    # 2. de_domega = cos_z * Gx * d(dTvec)/d(omega) 
    analytic_H2 = cos*Gx@np.kron(T.matrix().T,np.eye(3))@G    

    # Numerical Jacobian 
    numerical_H1 = numericalDerivative21(h, T, omega)
    numerical_H2 = numericalDerivative22(h, T, omega)

    assert np.allclose(numerical_H1,analytic_H1) and np.allclose(numerical_H2,analytic_H2)


def test_constraint_factor_adjoint():
    # Section IV.C 
    Gx = np.array([[0,0,0,0,0,0,0,0,0,1,0,0]])
    Gtheta = np.array([[0,0,1,0,0,0]])
    G  = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0. , 0.,  1.,  0.,  0.,  0.],
                   [ 0. ,-1.,  0.,  0.,  0.,  0.],
                   [ 0. , 0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0., -0.],
                   [ 1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]])
    
    T_bc = np.array([[ 0, 0, 1,0.9],
                     [-1, 0, 0,  0],
                     [ 0,-1, 0,  0],
                     [ 0, 0, 0,  1]])
    T_bc = Pose3(T_bc)

    def vec(M:np.ndarray):
        """Vectorize top 3x4 block (12x1), column-major"""
        return M[0:3,:].reshape(-1, order='F')

    def h(pose:Pose3, omega:np.ndarray):
        omega_wb = T_bc.Adjoint(omega)
        T_wb = T_bc.matrix()@pose.matrix()@T_bc.inverse().matrix()
        
        T_dot_wb = Pose3.Hat(omega_wb) @ T_wb
        T_dot_wb_vec = vec(T_dot_wb)
        
        theta = np.dot(Gtheta,Pose3.Logmap(Pose3(T_wb)))
        return np.array([np.cos(theta)@Gx@T_dot_wb_vec])

    # Pose
    R = Rot3.Rodrigues(0.3,0.3,0.3) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t) 
    T_wb = T_bc.matrix()@T.matrix()@T_bc.inverse().matrix()
    T_wb = Pose3(T_wb)

    # Omega (generalized velocity)
    rho = np.array([10.0, -0.2, 1.5])   
    phi = np.array([0.01, np.pi/8, -0.03]) 
    omega = np.hstack((phi, rho)) 
    omega_wb = Pose3.AdjointMap(Pose3(T_bc))@omega

    # Yaw angle and Delta Pose
    theta = np.dot(Gtheta,Pose3.Logmap(T_wb))
    sin = np.sin(theta)
    cos = np.cos(theta)
    Tdot = Pose3.Hat(omega_wb) @ T_wb.matrix()
    Tdot_vec = vec(Tdot)

    # Analytic Jacobian
    AdjTbc = T_bc.AdjointMap()

    Jrinv_Twb = Pose3.LogmapDerivative(T_wb)
    Jrinv_Twc = Pose3.LogmapDerivative(T)
    Jl_AdjTbc_T = Pose3.ExpmapDerivative(AdjTbc@Pose3.Logmap(T))
    
    dDTvec_dTvec = np.kron(np.eye(4), Pose3.Hat(omega_wb)[:3,:3])
    dTvec_dTwb = np.kron(np.eye(4), T_wb.rotation().matrix()) @ G
    dTwb_dTwc = Jl_AdjTbc_T@AdjTbc@Jrinv_Twc

    # 1. Chain rule
    # 1.1 d(cos(theta))/d(T) * Gx * Tdot_vec  
    temp1 = -sin*Gx@Tdot_vec*Gtheta@Jrinv_Twb@dTwb_dTwc
    # 1.2 cos(theta) * Gx * d(Tdot_vec)/d(T)
    temp2 = cos@Gx@dDTvec_dTvec@dTvec_dTwb@dTwb_dTwc
    
    analytic_H1 = temp1 + temp2
    
    # 2. de_domega = de_dTdot * dTdot_domega * domega_b_domega_c 
    de_dTdot = cos*Gx
    dTdot_domega = np.kron(T_wb.matrix().T,np.eye(3))@G
    domegab_domegac = AdjTbc
    analytic_H2 = de_dTdot@dTdot_domega@domegab_domegac

    # Numerical Jacobian
    numerical_H1 = numericalDerivative21(h, T, omega)
    numerical_H2 = numericalDerivative22(h, T, omega)
    
    assert np.allclose(analytic_H1,numerical_H1) and np.allclose(analytic_H2,numerical_H2)
    
def test_constraint_factor_adjoint_right_increment():
    # Section IV.C 
    Gx = np.array([[0,0,0,0,0,0,0,0,0,1,0,0]])
    Gtheta = np.array([[0,0,1,0,0,0]])
    G  = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0. , 0.,  1.,  0.,  0.,  0.],
                   [ 0. ,-1.,  0.,  0.,  0.,  0.],
                   [ 0. , 0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0., -0.],
                   [ 1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]])
    
    T_bc = np.array([[ 0, 0, 1,0.9],
                     [-1, 0, 0,  0],
                     [ 0,-1, 0,  0],
                     [ 0, 0, 0,  1]])
    T_bc = Pose3(T_bc)

    def vec(M:np.ndarray):
        """Vectorize top 3x4 block (12x1), column-major"""
        return M[0:3,:].reshape(-1, order='F')

    def h(pose:Pose3, omega:np.ndarray):
        omega_wb = T_bc.Adjoint(omega)
        T_wb = T_bc.matrix()@pose.matrix()@T_bc.inverse().matrix()
        
        T_dot_wb =  T_wb @ Pose3.Hat(omega_wb)
        T_dot_wb_vec = vec(T_dot_wb)
        
        theta = np.dot(Gtheta,Pose3.Logmap(Pose3(T_wb)))
        return np.array([np.cos(theta)@Gx@T_dot_wb_vec])

    # Pose
    R = Rot3.Rodrigues(0.3,0.3,0.3) 
    t = Point3(3.5,-2.2,4.2)      
    T = Pose3(R,t) 
    T_wb = T_bc.matrix()@T.matrix()@T_bc.inverse().matrix()
    T_wb = Pose3(T_wb)

    # Omega (generalized velocity)
    rho = np.array([10.0, -0.2, 1.5])   
    phi = np.array([0.01, np.pi/8, -0.03]) 
    omega = np.hstack((phi, rho)) 
    omega_wb = Pose3.AdjointMap(Pose3(T_bc))@omega

    # Yaw angle and Delta Pose
    theta = np.dot(Gtheta,Pose3.Logmap(T_wb))
    sin = np.sin(theta)
    cos = np.cos(theta)
    Tdot = T_wb.matrix() @ Pose3.Hat(omega_wb)
    Tdot_vec = vec(Tdot)

    # Analytic Jacobian
    AdjTbc = T_bc.AdjointMap()

    Jrinv_Twb = Pose3.LogmapDerivative(T_wb)
    Jrinv_Twc = Pose3.LogmapDerivative(T)
    Jl_AdjTbc_T = Pose3.ExpmapDerivative(AdjTbc@Pose3.Logmap(T))
    
    dDTvec_dTvec = np.kron(Pose3.Hat(omega_wb).T, np.eye(3))
    dTvec_dTwb = np.kron(np.eye(4), T_wb.rotation().matrix()) @ G
    dTwb_dTwc = Jl_AdjTbc_T@AdjTbc@Jrinv_Twc

    # 1. Chain rule
    # 1.1 d(cos(theta))/d(T) * Gx * Tdot_vec  
    temp1 = -sin*Gx@Tdot_vec*Gtheta@Jrinv_Twb@dTwb_dTwc
    # 1.2 cos(theta) * Gx * d(Tdot_vec)/d(T)
    temp2 = cos@Gx@dDTvec_dTvec@dTvec_dTwb@dTwb_dTwc
    
    analytic_H1 = temp1 + temp2
    
    # 2. de_domega = de_dTdot_vec * dTdot_vec_domega * domega_b_domega_c 
    de_dDTvec = cos*Gx
    dDTvec_domega = np.kron(np.eye(4),T_wb.rotation().matrix())@G
    domegab_domegac = AdjTbc
    analytic_H2 = de_dDTvec@dDTvec_domega@domegab_domegac

    # Numerical Jacobian
    numerical_H1 = numericalDerivative21(h, T, omega)
    numerical_H2 = numericalDerivative22(h, T, omega)
    
    assert np.allclose(analytic_H1, numerical_H1) and np.allclose(analytic_H2,numerical_H2)

def test_vectorized_pose():
    # Section IV.C     
    # This G matrix has two tweaks for proper use with
    # vectorized poses 12 x 1 and also GTSAM where rotations and position 
    G  = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0. , 0.,  1.,  0.,  0.,  0.],
                   [ 0. ,-1.,  0.,  0.,  0.,  0.],
                   [ 0. , 0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0., -0.],
                   [ 1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]])

    def vec(M:np.ndarray):
        """Vectorize top 3x4 block (12x1), column-major"""
        return M[0:3,:].reshape(-1, order='F')

    def h(pose1: Pose3, pose2: Pose3):
        return vec(pose1.matrix()@pose2.matrix())
    
    # Values from testPose3.cpp
    R = Rot3.RzRyRx(np.pi/3, np.pi/3, np.pi/3)
    t = Point3(0.5, -0.2, 0.1)
    T1 = Pose3(R, t)

    R = Rot3.RzRyRx(0,0,np.pi/2)
    t = Point3(0.1, 0.5, -0.5)
    T2 = Pose3(R, t)

    analytic_H1 = np.kron(T2.matrix().T, T1.rotation().matrix()) @ G
    numerical_H1 = numericalDerivative21(h, T1, T2)
    
    assert np.allclose(numerical_H1,analytic_H1)


def test_vectorized_dot_T():
    # Section IV.C     
    # This G matrix has two tweaks for proper use with
    # vectorized poses 12 x 1 and also GTSAM where rotations and position 
    G  = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0. , 0.,  1.,  0.,  0.,  0.],
                   [ 0. ,-1.,  0.,  0.,  0.,  0.],
                   [ 0. , 0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0., -0.],
                   [ 1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]])

    def vec(M:np.ndarray):
        """Vectorize top 3x4 block (12x1), column-major"""
        return M[0:3,:].reshape(-1, order='F')

    def h(omega:np.ndarray,pose:Pose3):
        return vec(Pose3.Hat(omega) @ pose.matrix())

    # Pose
    R = Rot3.RzRyRx(np.pi/3, np.pi/3, np.pi/3)
    t = Point3(0.5, -0.2, 0.1)
    T1 = Pose3(R, t)

    # Generalized velocity vector 
    rho = np.array([10.0, -0.2, 1.5])   
    phi = np.array([0.01, np.pi/8, -0.03])
    omega = np.hstack((phi, rho))          

    analytic_H1 = np.kron(T1.matrix().T,np.eye(3))@G
    numerical_H1 = numericalDerivative21(h, omega,T1)

    dDTvec_dTvec = np.kron(np.eye(4), Pose3.Hat(omega)[:3,:3])
    dTvec_dxi = np.kron(np.eye(4), T1.rotation().matrix()) @ G

    analytic_H2 = dDTvec_dTvec @ dTvec_dxi
    numerical_H2 = numericalDerivative22(h, omega,T1)

    assert np.allclose(numerical_H1,analytic_H1) and np.allclose(numerical_H2,analytic_H2)


def test_vectorized_dot_T_right_increment():
    # This factor deviates from the dot_T as now the formula is 
    # \dot{T}= T * Pose.Hat(omega) in order to exploit the fact
    # that the Pose spline utilizes the local omega (associated to the right increments)

    # This G matrix has two tweaks for proper use with
    # vectorized poses 12 x 1 and also GTSAM where rotations and position 
    G  = np.array([[ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0. , 0.,  1.,  0.,  0.,  0.],
                   [ 0. ,-1.,  0.,  0.,  0.,  0.],
                   [ 0. , 0., -1.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0., -0.],
                   [ 1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  1.,  0.,  0.,  0.,  0.],
                   [-1.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  0.],
                   [ 0.,  0.,  0.,  1.,  0.,  0.],
                   [ 0.,  0.,  0.,  0.,  1.,  0.],
                   [ 0.,  0.,  0.,  0.,  0.,  1.]])

    def vec(M:np.ndarray):
        """Vectorize top 3x4 block (12x1), column-major"""
        return M[0:3,:].reshape(-1, order='F')

    def h(omega:np.ndarray,pose:Pose3):
        return vec(pose.matrix() @ Pose3.Hat(omega))

    # Pose
    R = Rot3.RzRyRx(np.pi/3, np.pi/3, np.pi/3)
    t = Point3(0.5, -0.2, 0.1)
    T = Pose3(R, t)

    # Generalized velocity vector 
    rho = np.array([10.0, -0.2, 1.5])   
    phi = np.array([0.01, np.pi/8, -0.03])
    omega = np.hstack((phi, rho))          

    analytic_H1 = np.kron(np.eye(4),T.rotation().matrix())@G
    numerical_H1 = numericalDerivative21(h, omega,T)

    dDTvec_dTvec = np.kron(Pose3.Hat(omega).T, np.eye(3))
    dTvec_dxi = np.kron(np.eye(4), T.rotation().matrix()) @ G

    analytic_H2 = dDTvec_dTvec @ dTvec_dxi
    numerical_H2 = numericalDerivative22(h, omega,T)

    assert np.allclose(numerical_H1,analytic_H1) and np.allclose(numerical_H2, analytic_H2)


if __name__ == "__main__":
    test_reproj_err_jacobian()
    test_object_motion_jacobian()
    test_vectorized_pose()
    test_vectorized_dot_T()
    test_constraint_factor()
    test_constraint_factor_adjoint()
    test_vectorized_dot_T_right_increment()