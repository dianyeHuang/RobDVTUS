'''
Author: Dianye Huang
Date: 2022-10-11 15:39:56
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-01-04 19:08:15
Description: 
    This script implements utilities for the 6D path fitting
'''

import numpy as np
import tf.transformations as t
from dmp_utils import TruncGaussApprox, Quaternion, AL_OrtDMP, AL_PosDMP
from math import acos, cos, sin


class RUSS_DMP:
    def __init__(self, posdmp:AL_PosDMP, ortdmp:AL_OrtDMP):
        self.total_dist = 0
        self.pos_dmp = posdmp
        self.ort_dmp = ortdmp
    
    def get_path_length(self):
        return self.total_dist
    
    def get_scan_poses(self, positions, prob_axis_arr, flag_diff=False):
        path_list = list()
        if flag_diff:
            positions.append(positions[-1])
            x_axis_arr_tmp = np.diff(np.array(positions), axis=0).reshape(-1, 3)
        else:
            x_axis_arr_tmp = positions
            
        for i in range(0, x_axis_arr_tmp.shape[0]):   # filter the 0 norm direction
            if np.linalg.norm(x_axis_arr_tmp[i]) < 1e-6:
                x_axis_arr_tmp[i] = x_axis_arr_tmp[i-1]
        
        x_axis_arr_norm = np.linalg.norm(x_axis_arr_tmp, axis=1).reshape(-1, 1) 
        x_axis_arr_norm = x_axis_arr_tmp / x_axis_arr_norm
        idx = 0
        for x_tmp, z_axis in zip(x_axis_arr_norm, prob_axis_arr):
            y_axis = np.cross(z_axis, x_tmp)
            x_axis = np.cross(y_axis, z_axis)
            y_axis = y_axis/np.linalg.norm(y_axis)
            x_axis = x_axis/np.linalg.norm(x_axis)    
            Rtmp = np.vstack((x_axis, y_axis, z_axis)).transpose()
            rot_mat = np.identity(4)
            rot_mat[:3,:3] = Rtmp
            quat = t.quaternion_from_matrix(rot_mat)
            quat = quat/np.linalg.norm(quat)
            pose = list(np.array(positions[idx]).reshape(-1)) + list(quat)
            idx += 1
            path_list.append(pose)
        return path_list
    
    def get_pos_diff(self, s_arr, pos_dmp:AL_PosDMP):
        new_s_arr = s_arr.copy()
        new_s_arr = new_s_arr + 0.0005/self.total_dist
        for i in range(new_s_arr.shape[0]):
            if new_s_arr[i] > 0:
                new_s_arr[i] = 1.0
        new_pos = pos_dmp.rollout(s_arr)
        new_pos2 = pos_dmp.rollout(new_s_arr)
        pos_diff = np.array(new_pos2) - np.array(new_pos)
        return pos_diff
    
    def fit(self, pos_list, probaxis_arr, sampleHz=30, vel_lim=0.01, flag_vis_result=False, pose_method='1'):
        self.total_dist, s_arr, pos_arr = self.pos_dmp.fit( sampleHz=sampleHz, position_list=pos_list, 
                                                            flag_visresult=flag_vis_result, vel_lim=vel_lim)
        pos_diff = self.get_pos_diff(s_arr, self.pos_dmp)
        if pose_method == '1':   # direct result
            path_list = get_scan_poses(pos_diff, probaxis_arr, flag_diff=False)
        elif pose_method == '2': # keep out of plane the same
            path_list = get_scan_poses2(pos_diff, probaxis_arr, flag_diff=False)
        elif pose_method == '3': # keep out of plane and in-plane the same
            z_axis_tmp = probaxis_arr
            mean_z_axis = np.mean(z_axis_tmp, axis=0).reshape(1, 3)
            mean_z_axis = mean_z_axis/np.linalg.norm(mean_z_axis)
            z_axis = np.ones(z_axis_tmp.shape[:2]) * mean_z_axis
            path_list = get_scan_poses(pos_diff, z_axis, flag_diff=False)
        else:
            print('no default method !!!')
        quat_arr = np.array(path_list)[:, 3::]    
        eQ_arr, quat_arr = self.ort_dmp.fit(s_arr, list(quat_arr), pos_list=pos_list, sampleHz=sampleHz, 
                                            vel_lim=vel_lim, flag_visresult=flag_vis_result, flag_check_quat=True)
        
        if flag_vis_result: 
            repro_quat_arr = self.ort_dmp.rollout(s_arr, flag_get_quat=True)
            plt_time =np.array(range(len(pos_arr)))/sampleHz
            plt.figure('Quaternion encoding')
            plt.plot(plt_time, quat_arr, linestyle='--', color='black')
            plt.plot(plt_time, repro_quat_arr, color='blue')
            plt.xlabel('Time (s)')
            plt.ylabel('Magnitude (rad)')
            plt.grid()
            plt.show()
        return s_arr

    def forward(self, s):
        pos = self.pos_dmp.forward(s)
        quat = self.ort_dmp.forward(s)
        return pos.reshape(-1), quat
        
    def rollout(self, s_arr):
        # output px, py, pz, qx, qy, qz, qw
        pos_arr = self.pos_dmp.rollout(s_arr)
        quat_arr = self.ort_dmp.rollout(s_arr, flag_get_quat=True)
        return pos_arr, quat_arr  


def get_scan_poses(positions, prob_axis_arr, flag_diff=False):
    path_list = list()
    if flag_diff:
        positions.append(positions[-1])
        x_axis_arr_tmp = np.diff(np.array(positions), axis=0).reshape(-1, 3)
    else:
        x_axis_arr_tmp = positions

    for i in range(0, x_axis_arr_tmp.shape[0]):
        if np.linalg.norm(x_axis_arr_tmp[i]) < 1e-6:
            x_axis_arr_tmp[i] = x_axis_arr_tmp[i-1]
    
    x_axis_arr_norm = np.linalg.norm(x_axis_arr_tmp, axis=1).reshape(-1, 1) # the last one is zero
    x_axis_arr_norm = x_axis_arr_tmp / x_axis_arr_norm
    idx = 0
    x_last = None
    for x_tmp, z_axis in zip(x_axis_arr_norm, prob_axis_arr):
        y_axis = np.cross(z_axis, x_tmp)
        x_axis = np.cross(y_axis, z_axis)
        y_axis = y_axis/np.linalg.norm(y_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)    
        Rtmp = np.vstack((x_axis, y_axis, z_axis)).transpose()
        rot_mat = np.identity(4)
        rot_mat[:3,:3] = Rtmp
        quat = t.quaternion_from_matrix(rot_mat)
        quat = quat/np.linalg.norm(quat)
        pose = list(np.array(positions[idx]).reshape(-1)) + list(quat)
        idx += 1
        path_list.append(pose)
    return path_list

def get_scan_poses2(positions, prob_axis_arr, flag_diff=False):
    # fixed out-of-plane orientation and change in-plane-orientation
    path_list = list()
    if flag_diff:
        positions.append(positions[-1])
        x_axis_arr_tmp = np.diff(np.array(positions), axis=0).reshape(-1, 3)
    else:
        x_axis_arr_tmp = positions

    for i in range(0, x_axis_arr_tmp.shape[0]):
        if np.linalg.norm(x_axis_arr_tmp[i]) < 1e-6:
            x_axis_arr_tmp[i] = x_axis_arr_tmp[i-1]
    
    x_axis_arr_norm = np.linalg.norm(x_axis_arr_tmp, axis=1).reshape(-1, 1) # the last one is zero
    x_axis_arr_norm = x_axis_arr_tmp / x_axis_arr_norm
    idx = 0
    x_last = None
    
    mean_z_axis = np.mean(prob_axis_arr, axis=0).reshape(1, 3) # TODO
    mean_z_axis = mean_z_axis / np.linalg.norm(mean_z_axis)
    
    for x_tmp, z_tmp in zip(x_axis_arr_norm, prob_axis_arr):
        y_axis = np.cross(z_tmp, x_tmp)
        x_axis = np.cross(y_axis, mean_z_axis)
        z_axis = np.cross(x_axis, y_axis)
        
        x_axis = x_axis/np.linalg.norm(x_axis) 
        y_axis = y_axis/np.linalg.norm(y_axis)
        z_axis = z_axis/np.linalg.norm(z_axis)
        
        Rtmp = np.vstack((x_axis, y_axis, z_axis)).transpose()
        rot_mat = np.identity(4)
        rot_mat[:3,:3] = Rtmp
        quat = t.quaternion_from_matrix(rot_mat)
        quat = quat/np.linalg.norm(quat)
        pose = list(np.array(positions[idx]).reshape(-1)) + list(quat)
        idx += 1
        path_list.append(pose)
    return path_list

import pickle
import matplotlib.pyplot as plt

import rospy
from geometry_msgs.msg import PoseArray, Pose


def get_posearray_msg(frame_id, pos_arr, quat_arr):
    posearr_msg = PoseArray()
    posearr_msg.header.frame_id = frame_id
    
    for pos, quat in zip(pos_arr, quat_arr):
        pose = Pose()
        pose.position.x = pos[0]
        pose.position.y = pos[1]
        pose.position.z = pos[2]
        pose.orientation.x = quat[0]
        pose.orientation.y = quat[1]
        pose.orientation.z = quat[2]
        pose.orientation.w = quat[3]
        posearr_msg.poses.append(pose)
    
    return posearr_msg

if __name__ == '__main__':
    picklepath = '/home/hdy/franka_russ_ws/src/path_plan/franka_path/tmp/demo_dict.pickle'
    with open(picklepath, 'rb') as f:
        pose_dict = pickle.load(f)
    print(type(pose_dict))
    print(pose_dict.keys())
    positions = np.array(pose_dict['scan positions'])[:-1:, :].reshape(-1, 3) 
    
    s_arr = np.linspace(0, 1, 400)
    z_axis = np.array(pose_dict['scan prob_axis'])
    
    pos_dmp = AL_PosDMP(n_kfns=41, th=3.0, k=3.5)
    ort_dmp = AL_OrtDMP(n_kfns=61, th=3.0, k=3.5)
    pose_dmp = RUSS_DMP(pos_dmp, ort_dmp)
    pose_dmp.fit(positions, z_axis, sampleHz=40, flag_vis_result=True, pose_method='1',vel_lim=0.005)
    
    pos_arr, quat_arr  = pose_dmp.rollout(s_arr)
    print('done1!')
    plt.figure(11)
    plt.plot(quat_arr)
    plt.grid()
    