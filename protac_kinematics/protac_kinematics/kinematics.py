import modern_robotics as mr

import tf_transformations as tf
import numpy as np

from ament_index_python.packages import get_package_share_directory
import os
import yaml


np.set_printoptions(precision=5, suppress=True)

class Kinematics():
    def __init__(self):

        kinematics_params_dir = os.path.join(get_package_share_directory('protac_kinematics'),
                                             'config',
                                             'kinematics_params.yaml'
                                             )
        with open(kinematics_params_dir) as file:
            self.kinematics_params_ = yaml.full_load(file)
    

        # Rotation matrix between base and world frame
        self.Rws = np.array(self.kinematics_params_['Rws']).reshape((3, 3))
        # initial effector pose w.r.t space frame
        self.M_ = np.array(self.kinematics_params_['init_pose']).reshape((4, 4))
        # Blist: the joint screw axes in the end-effector frame when the protac is
        # at home position
        self.Blist_ = np.array(self.kinematics_params_['screw_axis_body_frame']).reshape((6, 4))
        # Slist: the joint screw axes in the space frame when the protac is
        # at home position
        self.Slist_ = np.array(self.kinematics_params_['screw_axis_space_frame']).reshape((6, 4))
        # get other kinematics information
        self.upper_arm_length_ = self.kinematics_params_['upper_arm_length']
        self.forearm_arm_length_ = self.kinematics_params_['forearm_arm_length']
        self.elbow_offset_ = self.kinematics_params_['elbow_offset']
        self.base_shoulder_distance_ = self.kinematics_params_['base_shoulder_distance']
        self.joint1_range_ = self.kinematics_params_['joint1_range']
        self.joint2_range_ = self.kinematics_params_['joint2_range']
        self.joint3_range_ = self.kinematics_params_['joint3_range']
        self.joint4_range_ = self.kinematics_params_['joint4_range']

    def FKinBody(self, thetalist):
        return mr.FKinBody(self.M_, self.Blist_, thetalist)

    def FKinSpace(self, thetalist):
        return mr.FKinSpace(self.M_, self.Slist_, thetalist)

    def IKinBody(self, T, thetalist0, eomg=0.01, ev=0.001):
        """
        Input:
            T: the desired end-effector configuration
            thetalist0: An initial guess
            eomg: A small positive tolerance on the end-effector orientation error
            ev: A small positive tolerance on the end-effector linear positoin error
        Output:
            thetalist: Joint variables that achieve T winthin the specified tolerances
            success: A logical value where TRUE means that the function found a solution
        """
        return mr.IKinBody(self.Blist_, self.M_, T, thetalist0, eomg=eomg, ev=ev)

    def IKinSpace(self, T, thetalist0, eomg=0.01, ev=0.001):
        """
        Equivalent to IKinBody, except the joint screw axes are specified in the space frame.
        """
        return mr.IKinSpace(self.Slist_, self.M_, T, thetalist0, eomg=eomg, ev=ev)

    def IKinPos(self, p):
        """
        Inverse kinematics for 3-dof protac arm
        Input:
            p: the desired end-effector position configuration
        """
        P = np.array(p)
        a2 = self.upper_arm_length_
        a3 = self.forearm_arm_length_
        d1 = self.elbow_offset_
        px = P[0]
        py = P[1]
        pz = P[2] - self.base_shoulder_distance_

        theta1 = np.array([np.arctan2(d1, np.sqrt(px**2+py**2-d1**2)) + np.arctan2(py, px),
                           np.arctan2(d1, -np.sqrt(px**2+py**2-d1**2)) + np.arctan2(py, px)])      

        D = (px**2+py**2+pz**2-d1**2-a2**2-a3**2)/(2*a2*a3)

        theta3 = np.array([np.arctan2(-D, np.sqrt(1-D**2+1e-12)),
                           np.arctan2(-D, -np.sqrt(1-D**2+1e-12))])

        solution_13 = np.array([[theta1[0], theta3[0]],
                                [theta1[0], theta3[1]],
                                [theta1[1], theta3[0]],
                                [theta1[1], theta3[1]]])
        num = -a2*np.cos(solution_13[:, 1])*pz + (a2*np.sin(solution_13[:, 1])- a3)*(np.cos(solution_13[:, 0])*px+np.sin(solution_13[:, 0])*py)
        den = (a2*np.sin(solution_13[:, 1]) - a3)*pz + a2*np.cos(solution_13[:, 1])*(np.cos(solution_13[:, 0])*px+np.sin(solution_13[:, 0])*py)
        theta23 = np.arctan2(num, den)
        theta2 = theta23 - solution_13[:, 1]
        solution_1 = solution_13[:, 0]
        solution_2 = theta2 + np.pi/2
        solution_3 = -(solution_13[:, 1] + np.pi/2)
        solution = np.array([solution_1, solution_2, solution_3, np.zeros_like(solution_2)]).T

        # check for joint solution validation
        removed_solution_idx = list()
        for solution_idx, (sol_1, sol_2, sol_3) in enumerate(zip(solution_1, solution_2, solution_3)):
            if not (np.deg2rad(self.joint1_range_[0]) <= sol_1 <= np.deg2rad(self.joint1_range_[1])):
                sol_1_tf = sol_1 - 2*np.pi
                if not (np.deg2rad(self.joint1_range_[0]) <= sol_1_tf <= np.deg2rad(self.joint1_range_[1])):
                    removed_solution_idx.append(solution_idx)
                else:
                    solution[solution_idx, 0] = sol_1_tf
        
            if not (np.deg2rad(self.joint2_range_[0]) <= sol_2 <= np.deg2rad(self.joint2_range_[1])):
                sol_2_tf = sol_2 - 2*np.pi
                if not (np.deg2rad(self.joint2_range_[0]) <= sol_2_tf <= np.deg2rad(self.joint2_range_[1])):
                    if solution_idx not in removed_solution_idx:
                        removed_solution_idx.append(solution_idx)
                else:
                    solution[solution_idx, 1] = sol_2_tf

            if not (np.deg2rad(self.joint3_range_[0]) <= sol_3 <= np.deg2rad(self.joint3_range_[1])):
                sol_3_tf = sol_3 + 2*np.pi
                if not (np.deg2rad(self.joint3_range_[0]) <= sol_3_tf <= np.deg2rad(self.joint3_range_[1])):
                    if solution_idx not in removed_solution_idx:
                        removed_solution_idx.append(solution_idx)
                else:
                    solution[solution_idx, 2] = sol_3_tf
        
        valid_solution = np.delete(solution, removed_solution_idx, axis=0)
        if len(valid_solution) == 0:
            raise('The pose cannot be reached!')
        else:
            return valid_solution

    def FindClosestIksolution(self, p, qseed):
        """Find the closest inverse kinematics solution 
        relative to the seed value (qseed)

        :param p: the target eef position of protac arm
        :param qseed: The seed joint values to which solutions compare against.
        :return: the closest ik solution to the seed.
        """

        solutions = self.IKinPos(p)
        if len(solutions) > 0:
            distances = [sum((qseed-qsol)**2) for qsol in solutions]
            closest = np.argmin(distances)
            return solutions[closest]
        else:
            return None

    def JacobianBody(self, thetalist):
        return mr.JacobianBody(self.Blist_, thetalist)

    def JacobianSpace(self, thetalist):
        return mr.JacobianSpace(self.Slist_, thetalist)

    def CubicTimeScaling(self, Tf, t):
        """Computes s(t) for a cubic time scaling
        
        :param Tf: Total time of the motion in seconds from rest to rest
        :param t: The current time t satisfying 0 < t < Tf
        :return: The path parameter s(t) and sdot(t) corresponding to a third-order
                polynomial motion that begins and ends at zero velocity
        """
        return mr.CubicTimeScaling(Tf, t), (6.0 * t) / Tf**2 - (6.0 * t**2) / Tf**3

    def QuinticTimeScaling(self, Tf, t):
        """Computes s(t) for a quintic time scaling
        
        :param Tf: Total time of the motion in seconds from rest to rest
        :param t: The current time t satisfying 0 < t < Tf
        :return: The path parameter s(t) corresponding to a third-order
                polynomial motion that begins and ends at zero velocity
        """
        return mr.QuinticTimeScaling(Tf, t), (30.0 * t**2) / Tf**3 - (60.0 * t**3) / Tf**4 + (30.0 * t**4) / Tf**5

    def LineTrajectoryGeneration(self, thetastart, thetaend, Tf, N, method):
        """Computes a straight-line trajectory in both joint and task space
        
        :param thetastart: The initial joint variables or 3D end-effector coordinate
        :param thetaend: The final joint variables of 3D end-effector coordinate
        :param Tf: Total time of the motion in seconds from rest to rest
        :param method: The time-scaling method, where 3 indicates cubic (third-
                    order polynomial) time scaling and 5 indicates quintic
                    (fifth-order polynomial) time scaling
        :return: A trajectory as an N x n matrix, where each row is an n-vector
                of joint variables at an instant in time. The first row is
                thetastart and the Nth row is thetaend . The elapsed time
                between each row is Tf / (N - 1)
        """
        N = int(N)
        timegap = Tf / (N - 1.0)
        traj = np.zeros((len(thetastart), N))
        trajdot = np.zeros((len(thetastart), N))
        for i in range(N):
            if method == 3:
                s, sdot = self.CubicTimeScaling(Tf, timegap * i)
            else:
                s, sdot = self.QuinticTimeScaling(Tf, timegap * i)
            traj[:, i] = s * np.array(thetaend) + (1 - s) * np.array(thetastart)
            trajdot[:, i] = sdot * (np.array(thetaend) - np.array(thetastart))
        traj = np.array(traj).T
        trajdot = np.array(trajdot).T
        return traj, trajdot 

def main():
    kin = Kinematics()
    Rws = kin.Rws
    # thetalist = np.deg2rad(np.array([-30, 10, -60, 0.]))
    thetalist = np.array([0.0, -1.308996939, -2.094395102, 0.0])
    # print(thetalist)

    T_eef = kin.FKinSpace(thetalist)
    # eef position in fixed space/base {s} frame
    ps = T_eef[:3, 3]
    # eef position in fixed world {w} frame
    pw = np.dot(Rws, ps)
    # P = [0.3, -0.2, 0.4]
    # pw_target = [0.223, 0.04, 0.459]
    pw_target = [0.705, -0.01,   -0.115]
    ps_target = np.dot(Rws.T, pw_target)
    # i_thetalist = kin.IKinPos(T_eef[:3, 3])
    i_thetalist = kin.IKinPos(ps_target)

    print('End-effector position in space {{s}} frame:\n {}'.format(ps))
    print('End-effector position in world {{w}} frame:\n {}'.format(pw))
    print('============================================================')
    print('Target end-effector position in space {{s}} frame:\n {}'.format(ps_target))
    print('Target end-effector position in world {{w}} frame:\n {}'.format(pw_target))
    print('============================================================')
    for theta in i_thetalist:
        print('Computed joint(deg): {}'.format(np.rad2deg(theta)))
        print('Computed joint(rad): {}'.format(theta))
        print('Check pose again:\n {}-{}'.format(kin.FKinSpace(theta), np.allclose(kin.FKinSpace(theta)[:3, 3], T_eef[:3, 3])))
