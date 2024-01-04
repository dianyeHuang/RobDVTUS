#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-08 14:23:56
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-01-04 19:09:28
Description: 

This script capture synchronized raw color images & aligned depth image,
and extract a path for franka to follow with.

us image size (height x width): 700 x 785
depth 4.5 cm
probe width 3.75 cm
downsample: 5 steps, final size: 140 x 157

'''

# ros packages
import rospy
import message_filters
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Float64MultiArray 
from std_msgs.msg import Bool

import tf
from tf import transformations as t

from std_msgs.msg import Header
from sensor_msgs import point_cloud2
from sensor_msgs.msg import PointCloud2, PointField

from geometry_msgs.msg import Point, PoseArray, Pose
from visualization_msgs.msg import Marker

# image packages
import os
import yaml
import cv2
from cv_bridge import CvBridge
import pyrealsense2 as rs
from skimage.morphology import skeletonize
import open3d as o3d


# others
import time
import numpy as np
import collections
import pickle
import matplotlib.pyplot as plt

from posort_dmp import RUSS_DMP
from dmp_utils import AL_OrtDMP, AL_PosDMP

from franka_controllers.utils.motion_rosinterface import MotionRosInterface, CtrlMode 


def path_planning_rosparam_init():
    rospy.set_param('/gui_franka/russ/detect_scan_path', False)
    rospy.set_param('/gui_franka/russ/start_scan', False)
    rospy.set_param('/gui_franka/russ/make_contact', False)
    rospy.set_param('/gui_franka/russ/move_forward', False)
    rospy.set_param('/gui_franka/russ/back_home', False)
    
    rospy.set_param('/gui_franka/russ/get_us_scanpath', False)
    rospy.set_param('/gui_franka/russ/repro_mkcnt', False)
    rospy.set_param('/gui_franka/russ/load_trajectory', False)

class Node:
    def __init__(self, data:list):
        self.left = None
        self.right = None
        self.data = data

class PathFilter:
    def __init__(self):
        pass
        
    def img_process(self, img, flag_show_num=False, flag_show_resimg=False, cimg=None):
        
        flag_ok = True
        final_path = list()
        
        if flag_show_num:
            num_list = list()
            for col in range(width):
                num = 0
                for row in range(height):
                    if img[row, col] > 0:
                        num += 1
                num_list.append(num)
            plt.figure(1)
            plt.plot(num_list)
            plt.grid()
            plt.show()
        
        (height, width) = img.shape
        start_col = -1
        start_row = -1
        for col in range(width):
            num = 0
            for row in range(height):
                if img[row, col] > 0:
                    num += 1
            if num == 1:
                start_col = col
                for row in range(height):
                    if img[row, start_col] > 0:
                        start_row = row
                break
        
        if start_col < 0 or start_row < 0:
            return final_path, False
        
        tree_root_node, vis_img1, flag_ok = self.get_path_tree(start_row, start_col, img, maxtime_line_detect=0.5)
        final_path = self.get_longest_path(tree_root_node)
        
        ret_img = np.zeros(img.shape, dtype=np.uint8)
        for p in final_path:
            ret_img[p[0], p[1]] = int(255)
        
        if flag_show_resimg:
            vis_img0 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            vis_img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            for pxy in final_path:
                vis_img2[pxy[0], pxy[1], :] = (0, 0, 255)
            
            # cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
            (height, width) = vis_img0.shape[:2]
            cimg = cv2.resize(cimg, (width, height))
            tmp_img0 = np.concatenate((cimg, vis_img0*255), axis=1)
            tmp_img1 = np.concatenate((vis_img1, vis_img2), axis=1)
            cv2.imshow('detect result', np.concatenate((tmp_img0, tmp_img1), axis=0))
            cv2.waitKey(2)
        
        return final_path, ret_img, flag_ok
    
    def get_longest_path(self, root):
        if root == None:
            return []
        rightvect = self.get_longest_path(root.right)
        leftvect = self.get_longest_path(root.left)
        if len(leftvect) > len(rightvect):
            leftvect.append(root.data[:2])
        else:
            rightvect.append(root.data[:2])
        
        if len(leftvect) > len(rightvect):
            return leftvect
        return rightvect
    
    def get_path_tree(self, start_row, start_col, img, maxtime_line_detect = 0.5):
        flag_ok = True
        
        vis_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        root_node = Node([start_row, start_col, 10])
        node_list = [root_node]
        
        CODE_OFFSET = 2
        CODE_NUM = 5
        
        start_time = rospy.get_time()
        while True:
            current_time = rospy.get_time()
            if current_time - start_time > maxtime_line_detect:
                flag_ok = False
                break
            
            tmp_node_list = list()
            for node in node_list:
                current_search_pt = node.data[:2]
                
                point_list = list()
                for idx in range(CODE_NUM):
                    if idx == node.data[2]:
                        continue
                    row, col = self.get_vincity_idx(current_search_pt, idx + CODE_OFFSET)
                    if img[row, col] > 0:
                        if idx == 0:
                            point_list.append([row, col, 4])
                        elif idx == 4:
                            point_list.append([row, col, 0])
                        else:
                            point_list.append([row, col, 10])
                            
                if len(point_list) == 1:
                    tmp_node_list.append(Node(point_list[0]))
                    node.right = tmp_node_list[-1]
                    vis_img[point_list[0][0], point_list[0][1], :] = (0, 255, 0)
                
                if len(point_list) == 2:
                    tmp_node_list.append(Node(point_list[0]))
                    node.right = tmp_node_list[-1]
                    vis_img[point_list[0][0], point_list[0][1], :] = (0, 255, 0)
                    tmp_node_list.append(Node(point_list[1]))
                    node.left = tmp_node_list[-1]
                    vis_img[point_list[1][0], point_list[1][1], :] = (0, 255, 0)
                        
            if len(tmp_node_list) == 0:
                break
            else:
                node_list = tmp_node_list
                
        return root_node, vis_img, flag_ok
    
    def get_vincity_idx(self, center_xy, n_code, flag_col_row=False):
        CODE_SEQ_ARR = [[-1,  0],  # 0
                        [-1,  1],  # 1
                        [ 0,  1],  # 2
                        [ 1,  1],  # 3
                        [ 1,  0],  # 4
                        [ 1, -1],  # 5
                        [ 0, -1],  # 6
                        [-1, -1]]  # 7
        CODE_MAX = 8
        n_code = n_code % CODE_MAX
        deviate_xy = CODE_SEQ_ARR[n_code]
        if flag_col_row:
            xidx = center_xy[0] + deviate_xy[0]
            yidx = center_xy[1] + deviate_xy[1]
        else:
            xidx = center_xy[0] + deviate_xy[1]
            yidx = center_xy[1] + deviate_xy[0]
        return [xidx, yidx] 

from geometry_msgs.msg import WrenchStamped
class IntentDetector:
    def __init__(self, rosnode_name=None):
        if rosnode_name is not None:
            rospy.init_node('intent_detect_controller_node', anonymous=True)
        
        self.force_est = None
        self.force_act = None
        self.force_cali = None
        self.scan_max_force = 15
        self.scan_min_force = 1.0
        
        self.estimate_force_sub  = rospy.Subscriber("/franka_state_controller/F_ext", 
                                        WrenchStamped, self.franka_force_est_cb, queue_size=1)
        
        self.measured_force_sub  = rospy.Subscriber("/franka_state_controller/Cali_F_ext",
                                        WrenchStamped, self.franka_force_cali_cb, queue_size=1)
        
        self.ctrl_info_sub = rospy.Subscriber("/physical_hri_controller/msg_ctrl_info", 
                                        Float64MultiArray, self.ctrl_info_cb, queue_size=1)
        
        self.des_force_pub = rospy.Publisher("/physical_hri_controller/desired_force",
                                        WrenchStamped, queue_size=1)
        
        rospy.wait_for_message("/franka_state_controller/F_ext", WrenchStamped)
        rospy.wait_for_message("/franka_state_controller/Cali_F_ext", WrenchStamped)
        
    def franka_force_est_cb(self, msg:WrenchStamped):
        self.force_est = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])

    def franka_force_cali_cb(self, msg:WrenchStamped):
        self.force_cali = np.array([
            msg.wrench.force.x,
            msg.wrench.force.y,
            msg.wrench.force.z
        ])
        
    def ctrl_info_cb(self, msg:Float64MultiArray):
        self.force_act = msg.data[0]  

    def path_interact(self, force_ext, total_dist, scale_factor, force_lim=4.0, deadzone=1.5):
        des_svel = 0
        if abs(force_ext) > deadzone:
            if force_ext > 0:
                if force_ext - deadzone > force_lim:
                    force_ext = force_lim
                else:
                    force_ext = force_ext - deadzone  # TODO
            else:
                if force_ext + deadzone < -force_lim:
                    force_ext = -force_lim
                else:
                    force_ext = force_ext + deadzone # TODO
            des_vel = force_ext*scale_factor # m/s
            des_svel = des_vel/total_dist    # unit w.r.t dist=1.0, note that s\in[0, 1] is dimensionless
        return des_svel
    
    def inter_force_clamp(self, force:float, limit:float):
        if force > limit:
            force = limit
        elif force < -limit:
            force = -limit
        return force
    
    def pub_des_force(self, des_force:float):  # TODO
        # publish target force
        desired_force_msg = WrenchStamped()
        desired_force_msg.wrench.force.z = des_force
        self.des_force_pub.publish(desired_force_msg)

from vesselscanpath import VesselScanPath
class FrankaPath:
    def __init__(self, pcd_savepath, pose_dmp:RUSS_DMP, loophz = 30):
        rospy.init_node('franka_path_planning', anonymous=True)
        self.pose_dmp = pose_dmp
        
        # global variables
        self.wait_update_img = True
        self.wTc = None
        self.point_positions = collections.OrderedDict()
        self.avg_num = 1
        self.avg_dimg_list = list()
        self.path_dimg = None # TODO
        
        self.path_filter = PathFilter()
        self.intent_detector = IntentDetector()
        self.vessel_scan = VesselScanPath()
        
        # others
        self.tf_listener = tf.TransformListener()
        self.br = tf.TransformBroadcaster()
        self.visual_img = Image()
        
        # realsense 
        self.bridgeC = CvBridge()
        self.mtx = np.array([[615.7256469726562, 0.0, 323.13262939453125], 
                            [0.0, 616.17236328125, 237.86715698242188], 
                            [0.0, 0.0, 1.0]])
        self.dist =np.array([ 0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0,0.0, 0.0, 0.0, 0.0])
        self._intrinsics = rs.intrinsics()
        
        # ros control communication
        # moveit class
        # controller switcher class
        self.loop_hz = loophz
        self.force_err = 0
        self.contact_alpha = 0
        self.ros_motion = MotionRosInterface(   pub_cmd_topic_name='/physical_hri_controller/desired_position', 
                                                loop_hz=self.loop_hz, max_vel_limit=0.2)
        self.sub_ctrl_info = rospy.Subscriber(  "/physical_hri_controller/msg_ctrl_info", Float64MultiArray,
                                                self.ctrl_info_cb, queue_size=1)
        
        self.footswitch_pressed = False
        self.sub_footsw = rospy.Subscriber( '/footswitch/flag_ispressing', Bool,
                                            self.foot_switcher_cb, queue_size=1)

        self.final_pose = self.ros_motion.get_posemat_from_tf_tree( child_frame='panda_EE', 
                                                                    parent_frame='panda_link0')
        print('current pose: ')
        print(self.final_pose)

        # subscribe images
        self.dimg_sub = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw",
                                                    Image, queue_size=20)                  # depth images
        self.cimg_sub = message_filters.Subscriber("/camera/color/image_raw", 
                                                    Image, queue_size=20, buff_size=2**24) # color images
        self.cinfo_sub = message_filters.Subscriber("/camera/color/camera_info", 
                                                    CameraInfo,queue_size=20 )             # intern parameters
        self.ts = message_filters.TimeSynchronizer([self.cimg_sub, self.dimg_sub, self.cinfo_sub], 1)
        self.ts.registerCallback(self.sub_imgs_cb)
        
        # point cloud settings
        self.pcd_savepath = pcd_savepath
        
        self.pub_point_cloud2 = rospy.Publisher('path_planning/skin/points', PointCloud2, queue_size=5) 
        self.pub_vis_path = rospy.Publisher('path_planning/skin/path', Marker, queue_size=1)
        self.pose_array = PoseArray()
        self.pose_array.header.frame_id = "panda_link0"
        self.pose_array.header.seq = 0
        self.pub_path_poses_raw = rospy.Publisher('path_planning/skin/path_poses_raw', PoseArray, queue_size=1)
        self.pub_path_poses_encode = rospy.Publisher('path_planning/skin/path_poses_encode', PoseArray, queue_size=1)
        self.pub_skin_skeleton_img = rospy.Publisher('path_planning/skin/skeleton_image', Image, queue_size=1)
        
        # wait for messages
        print('waiting for realsense topics ... ')
        rospy.wait_for_message("/camera/aligned_depth_to_color/image_raw", Image)
        rospy.wait_for_message("/camera/color/image_raw", Image)
        rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        print('all topics received ... ')
        
    def foot_switcher_cb(self, msg:Bool):
        self.footswitch_pressed = msg.data
        
    def ctrl_info_cb(self, msg):
        self.actual_force = msg.data[0]
        self.des_force = msg.data[1]
        self.force_err = abs(self.actual_force - self.des_force)
        self.contact_alpha = msg.data[2]
        # print('self.force_err: ', self.force_err)
        # print('self.contact_alpha: ', self.contact_alpha)

    def get_pcd_from_points(self, points, savepath):
        point_num = len(points)
        with open(savepath, 'w') as f:
            f.write('# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1')
            f.write('\nWIDTH ' + str(point_num))
            f.write('\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0')
            f.write('\nPOINTS ' + str(point_num))
            f.write('\nDATA ascii')
        with open(savepath, 'a') as f:
            for p in points:
                # f.write('\n' + str(p.x) + ' ' + str(p.y) + ' ' + str(p.z))
                f.write('\n' + str(p[0]) + ' ' + str(p[1]) + ' ' + str(p[2]))
        return o3d.io.read_point_cloud(savepath)
    
    def sub_imgs_cb(self, ros_cimg, ros_dimg, ros_cinfo):
        if self.wait_update_img:
            self.wait_update_img = False
            # get images
            self.load_cam_info(ros_cinfo)
            self.cv2_cimg = self.bridgeC.imgmsg_to_cv2(ros_cimg, "bgr8")
            cv2_dimg = self.bridgeC.imgmsg_to_cv2(ros_dimg, desired_encoding="passthrough")
            self.path_dimg = np.array(cv2_dimg, dtype=np.float32)   # for computing the path 
            
            if len(self.avg_dimg_list) >= self.avg_num:
                self.avg_dimg_list = self.avg_dimg_list[1:]
            self.avg_dimg_list.append(cv2_dimg)
            self.cv2_dimg = np.mean(np.array(self.avg_dimg_list), axis=0)  # for computing the norm vector
            
    def load_cam_info(self, ros_cinfo):
        self._intrinsics.width  = ros_cinfo.width
        self._intrinsics.height = ros_cinfo.height
        self._intrinsics.ppx = ros_cinfo.K[2]  # 2
        self._intrinsics.ppy = ros_cinfo.K[5]  # 5
        self._intrinsics.fx  = ros_cinfo.K[0]  # 0
        self._intrinsics.fy  = ros_cinfo.K[4]  # 4
        self._intrinsics.model  = rs.distortion.inverse_brown_conrady
        self._intrinsics.coeffs = [i for i in ros_cinfo.D]
    
    def get_tfmtx_from_tf_tree(self, child_frame, parent_frame):
        (trans,rot) = self.tf_listener.lookupTransform(parent_frame,   # parent frame
                                                        child_frame,   # child frame
                                                        rospy.Time(0))
        res_T = t.concatenate_matrices(t.translation_matrix(trans), 
                                        t.quaternion_matrix(rot))
        return res_T
    
    def save_pickle(self, d, savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(d, f)
    
    def get_and_encode_scan_path(self, seg_h, sstep, crop = None, flag_save_data=False, flag_vis_mid_result=False):
        # get starting and ending point limit
        centers, flag_detected = self.color_sticker_detection(self.cv2_cimg, sstep=sstep, vis_img=flag_vis_mid_result)
        if flag_detected:
            cols_lim = list(np.array(centers)[:, 0].reshape(-1))
            cols_lim.sort(reverse=False)
            # print('cols_lim:', cols_lim)
            if crop is not None and len(cols_lim) == 2: # TODO
                cols_lim[0] += crop
                if cols_lim[0] > self.cv2_cimg.shape[1] - 1:
                    cols_lim[0] = self.cv2_cimg.shape[1] - 1
                cols_lim[1] -= crop
                if cols_lim[1] < 0:
                    cols_lim[1] = 0
            else:
                print('stiker more than 2 or no stiker!')
                return False
        else:
            print('no color stiker detected!')
            return False
        
        # get skeletonized images
        skin_img, skin_bin, pc_list, _ = self.extract_skin_info(self.cv2_cimg, self.cv2_dimg, 
                                                                self._intrinsics, self.wTc, seg_h, sstep)
        # get scan path position
        self.path_positions, self.idx_list, flag_ok = self.get_scan_positions(skin_bin, cols_lim, flag_vis_img=flag_vis_mid_result)
        if not flag_ok:
            return False
        
        
        
        # get scan path probe axis 
        pcd = self.get_pcd_from_points(pc_list, self.pcd_savepath)
        pcd_search_param=o3d.geometry.KDTreeSearchParamRadius(radius=0.01)
        pcd.estimate_normals(search_param=pcd_search_param)
        
        self.probe_axis = np.asarray(pcd.normals)[self.idx_list, :] # surface normal direction
        # get scan path position and orientation
        # - check directioni consistency, the normal direction should be aligned with the camera view, pointing towards camera frame
        self.probe_axis = -self.check_normal_consistency(self.path_positions, self.probe_axis, self.wTc[:3, 3])
        
        # extract and encode scan path
        s_arr = self.pose_dmp.fit(self.path_positions, self.probe_axis, sampleHz=10, flag_vis_result=False, pose_method='3')   # self.pose_dmp is updated here
        pos_arr, quat_arr = self.pose_dmp.rollout(s_arr)
        self.path_pose_encode =  np.hstack((pos_arr, quat_arr))
        
        self.path_positions = self.path_positions[:-1:]
        self.path_pose_raw = self.get_scan_poses2(self.path_positions, self.probe_axis) 
        
        # path fitting using dmp
        # save demo
        if flag_save_data:
            save_dict = dict()
            save_pickle_path = self.pcd_savepath.replace('.pcd', '_dict.pickle')
            print('save_pickle_path: ', save_pickle_path)
            save_dict['scan positions'] = self.path_positions
            save_dict['scan prob_axis'] = self.probe_axis
            save_dict['scan path_pose_raw'] = self.path_pose_raw
            save_dict['scan path_pose_encode'] = self.path_pose_encode
            self.save_pickle(save_dict, save_pickle_path)
        
        
        # publish results
        pc_msg = self.get_point_cloud(pc_list, pc_frame_id = "panda_link0")
        self.pub_point_cloud2.publish(pc_msg)
        
        msg_marker = self.generate_markers(self.path_positions, 'panda_link0', 'img_path', [0, 0, 1, 1])
        self.pub_vis_path.publish(msg_marker)
        
        posearray_msg = self.get_posearray_msg(self.path_pose_raw)
        self.pub_path_poses_raw.publish(posearray_msg)
        
        posearray_msg = self.get_posearray_msg(self.path_pose_encode)
        self.pub_path_poses_encode.publish(posearray_msg)
        
        return True
    
    def pub_dmp_pose(self, pose_dmp:RUSS_DMP, num_points=200):
        s_arr = np.linspace(0, 1, num_points)
        pos_arr, quat_arr = pose_dmp.rollout(s_arr)
        path_pose_encode =  np.hstack((pos_arr, quat_arr))
        posearray_msg = self.get_posearray_msg(path_pose_encode)
        self.pub_path_poses_encode.publish(posearray_msg)
    
    def get_scan_poses(self, positions:list, prob_axis_arr:np.array):
        path_list = list()
        positions.append(positions[-1])
        x_axis_arr_tmp = np.diff(np.array(positions), axis=0).reshape(-1, 3)
        x_axis_arr_tmp[-1] = x_axis_arr_tmp[-2]
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
            pose = list(np.array(positions[idx]).reshape(-1)) + list(quat)
            idx += 1
            path_list.append(pose)
        return path_list
    
    def get_scan_poses2(self, positions, prob_axis_arr, flag_diff=False):
        # fixed out-of-plane orientation and change in-plane-orientation
        path_list = list()
        positions.append(positions[-1])
        x_axis_arr_tmp = np.diff(np.array(positions), axis=0).reshape(-1, 3)
        x_axis_arr_tmp[-1] = x_axis_arr_tmp[-2]
        x_axis_arr_norm = np.linalg.norm(x_axis_arr_tmp, axis=1).reshape(-1, 1)
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
    
    def get_posearray_msg(self, pose_list):
        self.pose_array.poses = []
        for p in pose_list:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = p[2]
            pose.orientation.x = p[3]
            pose.orientation.y = p[4]
            pose.orientation.z = p[5]
            pose.orientation.w = p[6]
            self.pose_array.poses.append(pose)
        self.pose_array.header.seq += 1
        self.pose_array.header.stamp = rospy.Time.now()
        return self.pose_array
            
    def check_normal_consistency(self, positions:np.array, probe_axis:np.array, camera_position:np.array):
        new_probe_axis = list()
        for pos, z_axis in zip(positions, probe_axis):
            if np.dot(camera_position - pos.reshape(-1), z_axis) > 0:
                new_probe_axis.append(list(z_axis))
            else:
                new_probe_axis.append(list(-z_axis))
        return np.array(new_probe_axis)

    def color_sticker_detection(self, img, sstep=1, vis_img=False,
                            low_range = np.array([0, 123, 100]), high_range = np.array([5, 255, 255])):
        flag_detected = True
        hue_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        th = cv2.inRange(hue_image, low_range, high_range)
        dilated = cv2.morphologyEx(th, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        dilated = cv2.erode(dilated, None, iterations=2)
        dilated = cv2.dilate(dilated, None, iterations=2)
        
        (cnts, _) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        closed_cs = sorted(cnts, key=cv2.contourArea, reverse=True)
        
        centers = list()
        for c in closed_cs:
            rect = cv2.minAreaRect(c)
            box = np.int0(cv2.boxPoints(rect))
            center_arr = np.mean(np.array(box), axis=0).reshape(-1)
            center = list(center_arr.astype(np.uint32))
            cv2.circle(img, center, 5, (0, 255, 0), 2)
            centers.append(list((center_arr/sstep).astype(np.uint32)))
        
        if len(centers)==0:
            flag_detected = False
        
        if vis_img:
            cv2.imshow('original_image', img)
            cv2.waitKey(1)
            cv2.imshow('color_tie_detect', dilated)
            cv2.waitKey(1)
        
        return centers, flag_detected
    
    def get_skeleton_img(self, img):
        kernel = np.ones((5,5),np.uint8)                     # hyperparameters to be determined
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)  # errode first and then expand
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel) 
        _, img_tmp = cv2.threshold(img, 128, 1, cv2.THRESH_BINARY)
        img = skeletonize(img_tmp, method='lee')
        return img
    
    def get_scan_positions(self, img, cols_lim, flag_vis_img=False):        
        skeleton_img = self.get_skeleton_img(img)
        _, ret_img, flag_ok = self.path_filter.img_process(skeleton_img, flag_show_resimg=flag_vis_img, cimg=self.cv2_cimg)
        self.pub_skin_skeleton_img.publish(self.bridgeC.cv2_to_imgmsg(ret_img))
        
        (rows, cols) = ret_img.shape
        path = list()
        idx_list = list()
        if len(cols_lim) >= 2:
            col_min, col_max = cols_lim[0], cols_lim[1]
        else:
            col_min, col_max = 0, cols
            
        for j in range(col_min, col_max):
            for i in range(rows):
                if ret_img[i,j]:   # filtered skeleton image
                    idx_str = str(i)+str(j)
                    path.append(self.point_positions[idx_str][:3:1])
                    idx_list.append(int(self.point_positions[idx_str][3, 0]))
        if len(path) < 3:
            print('<{}> length of the detected path: {}'.format(rospy.Time.now(), len(path)))
            flag_ok = False
        else:
            print('<{}> path detected ...'.format(rospy.Time.now()))
        return path, idx_list, flag_ok
    
    def generate_markers(self, traj_data, frame_str, ns_str, rgba_list):
        points = Marker()
        points.header.frame_id = frame_str
        points.header.stamp = rospy.Time.now()

        points.ns = ns_str
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0 # no orientation

        points.id = 0
        points.type = Marker.POINTS

        points.scale.x = 0.002   # define scales
        points.scale.y = 0.003
        points.scale.z = 0.002

        points.color.r = rgba_list[0]   # color 
        points.color.g = rgba_list[1]   
        points.color.b = rgba_list[2]
        points.color.a = rgba_list[3]

        for point in traj_data:
            p = Point()
            p.x = point[0]
            p.y = point[1]
            p.z = point[2]
            points.points.append(p)

        return points 
    
    def extract_skin_info(self, cimg, dimg, cinfo, wTcam, seg_height, sstep=1):
        '''
        extract skin image and point cloud based on the given color image and corresponding depth
        information. sstep means sample step and is for downsampling the image.
        the point cloud list is stored in rows
        '''
        (height, width, _) = cimg.shape
        cimg = cimg[0:height:sstep, 0:width:sstep]  # downsample image
        (height, width, _) = cimg.shape
        
        pc_list = list()
        color_list = list()
        self.point_positions = collections.OrderedDict() # TODO
        pc_stored_seq = 0
        skin_bin = np.zeros(cimg.shape[:2], dtype=np.uint8)
        for i in range(0, height):
            for j in range(0, width):
                tar_pixel = np.array([[j*sstep], [i*sstep], [1.0]])
                tmp = self.dimg_2_tras_vec([tar_pixel], dimg, cinfo)
                tmp.extend([1.0])
                c_T_tar_pos = np.array(tmp).reshape(4,1)
                pos = np.dot(wTcam, c_T_tar_pos)    # already has position of the points
                # print('pos: ', pos.shape)
                self.point_positions[str(i)+str(j)] = pos # log position info
                self.point_positions[str(i)+'_'+str(j)] = pos # log position info
                if c_T_tar_pos[2] < 0.10:
                    continue
                if pos[2] > seg_height:
                    skin_bin[i, j] = int(255)
                    pos[3] = pc_stored_seq
                    pc_stored_seq += 1
                    pc_list.append(list(pos.reshape(-1)))
                    color_list.append(list(cimg[i, j, :]))
        res = cv2.bitwise_and(cimg, cimg, mask=skin_bin)
        return res, skin_bin, pc_list, color_list

    def extract_pc_from_dimg(self, seg_img, dimg, sstep, cinfo, wTcam):
        pc_list = list()
        (height, width) = seg_img.shape
        for i in range(0, height):
            for j in range(0, width):
                if seg_img[i, j] == int(255):
                    tar_pixel = np.array([[j*sstep], [i*sstep], [1.0]])
                    tmp = self.dimg_2_tras_vec([tar_pixel], dimg, cinfo)
                    tmp.extend([1.0])
                    c_T_tar_pos = np.array(tmp).reshape(4,1)
                    pos = np.dot(wTcam, c_T_tar_pos)    # already has position of the points
                    pc_list.append(list(pos.reshape(-1)))
        return pc_list
    
    def dimg_2_tras_vec(self, ccorner, dimg_arr, cinfo):
        pixel2depth_marker = rs.rs2_deproject_pixel_to_point(cinfo, 
                                                            [int(ccorner[0][0]), 
                                                            int(ccorner[0][1])], 
                                                            dimg_arr[int(ccorner[0][1]),
                                                                    int(ccorner[0][0])])
        pixel2depth_marker[0] = float(pixel2depth_marker[0])/1000.0
        pixel2depth_marker[1] = float(pixel2depth_marker[1])/1000.0
        pixel2depth_marker[2] = float(pixel2depth_marker[2])/1000.0
        return pixel2depth_marker
    
    def get_point_cloud(self, pc_list, color_list=None, pc_frame_id = "camera_color_optical_frame"):
        if color_list is None:
            pc_array = np.array(pc_list)
            dtype = np.float32
            itemsize = np.dtype(dtype).itemsize
            data = pc_array.astype(dtype).tobytes()
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
            ]
        else:
            color_flatten = np.array(color_list, dtype=np.uint32)
            rgba_ch = (np.zeros((color_flatten.shape[0]), dtype=np.uint32)).reshape(-1, 1).astype(np.uint32) 
            rgba_ch = np.bitwise_or(rgba_ch, np.left_shift(color_flatten[:, 2], 16).reshape(-1, 1))
            rgba_ch = np.bitwise_or(rgba_ch, np.left_shift(color_flatten[:, 1], 8).reshape(-1, 1))
            rgba_ch = np.bitwise_or(rgba_ch, np.left_shift(color_flatten[:, 0], 0).reshape(-1, 1))
            pc_array = np.concatenate((np.array(pc_list)[:, :3], rgba_ch.reshape(-1, 1)), axis = 1)
            dtype = np.float32
            itemsize = np.dtype(dtype).itemsize
            data = pc_array.astype(dtype).tobytes()
            fields = [
                PointField('x', 0, PointField.FLOAT32, 1),
                PointField('y', 4, PointField.FLOAT32, 1),
                PointField('z', 8, PointField.FLOAT32, 1),
                PointField('rgb', 12, PointField.UINT32, 1)
            ]
        
        header = Header(frame_id=pc_frame_id, stamp=rospy.Time.now())
        return PointCloud2(
            header=header,
            height=1,
            width=pc_array.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 4),
            row_step=(itemsize * 4 * pc_array.shape[0]),
            data=data
        )
    
    def run(self):
        sstep = 4
        seg_h = 0.035
        flag_start_scan = False
        flag_learned_dmp = False
        flag_start_interact = False  # make contact
        flag_back_home = False
        flag_move_forward = False
        flag_reproduce = False
        flag_record_img_start = False
        flag_record_img_stop = False
        
        flag_adjust_force = False
        
        path_planning_rosparam_init()
        
        cmd_res = os.popen('rospack find franka_gui')
        self.franka_gui_dir = cmd_res.read()[:-1] # get rid of '\n'
        
        scan_state = 1
        # scan_vel = 0.030  # 7 mm/s  
        # scan_vel = 0.025    # 7 mm/s 
        scan_vel = 0.015    # 7 mm/s 
        # scan_vel = 0.005  # 3 mm/s 
        # scan_vel = 0.015    # 7 mm/s
        # scan_vel = 0.030    # 7 mm/s
        dmp_vel = 0
        self.dmp_s = 0
        init_upper_length = 0.03  # TODO
        final_upper_length = 0.03 # TODO
        
        rate = rospy.Rate(self.loop_hz)
        while not rospy.is_shutdown():
            rate.sleep()
            if not self.wait_update_img:
                self.wait_update_img = True
                if rospy.get_param('/gui_franka/russ/detect_scan_path'):  # detect path
                    self.wTc = self.get_tfmtx_from_tf_tree(child_frame='camera_color_optical_frame',
                                                            parent_frame='panda_link0')
                    flag_reproduce = False
                    flag_learned_dmp = False
                    flag_move_forward = False
                    flag_start_interact = False
                    if not self.get_and_encode_scan_path(seg_h, sstep, crop=5, flag_vis_mid_result=False, flag_save_data=True):
                        continue
                    dmp_vel = scan_vel/self.pose_dmp.get_path_length()
                    flag_learned_dmp = True
                    print('dmp model updated!')
                    
                    # save pose_dmp
                    save_path = '/home/hdy/Projects_ws/RobotAssistedDVT_tii/franka_russ_ws/src/path_plan/franka_path/tmp/tmp_pose_dmp.pkl' # TODO
                    with open(save_path, 'wb') as f:
                        pickle.dump([dmp_vel, self.pose_dmp], f)
                    print('save done!')
                    
                    rospy.set_param('/gui_franka/russ/detect_scan_path', False)
                    rospy.set_param('/gui_franka/russ/make_contact', False)
                
                if rospy.get_param('/gui_franka/russ/get_us_scanpath'):  # get final vessel path and visual
                    
                    filepath = '/home/hdy/Projects_ws/RobotAssistedDVT_tii/franka_russ_ws/src/path_plan/franka_networks/data/usimg_eepose_tmp.pickle' # TODO path of loading data
                    pcd_file = '/home/hdy/Projects_ws/RobotAssistedDVT_tii/franka_russ_ws/src/path_plan/franka_path/tmp/tmp.pcd'
                    
                    pose_dmp, scan_pose = self.vessel_scan.get_scan_path(filepath, pcd_file)
                    
                    self.pose_dmp = pose_dmp  # TODO dmp changed here
                    flag_learned_dmp = True
                    flag_reproduce = True
                    flag_move_forward = False
                    flag_start_interact = False
                    
                    # visualize results
                    self.vessel_scan.publish_final_posearr()
                    self.vessel_scan.publish_final_centerlines()
                    self.vessel_scan.publish_final_markers()
                    self.vessel_scan.publish_final_vessel()
                    
                    print('scan_pose shape: ', scan_pose.shape)
                    rospy.set_param('/gui_franka/russ/get_us_scanpath', False)
                    rospy.set_param('/gui_franka/russ/make_contact', False)
                
                if rospy.get_param('/gui_franka/russ/load_trajectory'):
                    rospy.set_param('/gui_franka/russ/load_trajectory', False)
                    print('load presaved pickle file!!')
                    filepath = '/home/hdy/Projects_ws/RobotAssistedDVT_tii/franka_russ_ws/src/path_plan/franka_path/tmp/tmp_pose_dmp.pkl'
                    with open(filepath, 'rb') as f:
                        res = pickle.load(f)
                    dmp_vel = res[0]
                    self.pose_dmp = res[1]
                    self.pub_dmp_pose(self.pose_dmp, num_points=200)
                    flag_learned_dmp = True
                
                if flag_learned_dmp:
                    if not flag_start_scan and rospy.get_param('/gui_franka/russ/start_scan'):   # begin scan 
                        rospy.set_param('/gui_franka/russ/start_scan', False)
                        
                        with open(self.franka_gui_dir + '/cfg/gui_franka.yaml') as file:
                            gui_cfg_params = yaml.load(file, Loader=yaml.FullLoader)
                        
                        # define middel pose during the scan
                        self.home_jnt_config = gui_cfg_params['gui_franka']['tab_russ']['home_jnt_config']
                        self.mid_jnt_config = gui_cfg_params['gui_franka']['tab_russ']['mid_jnt_config']
                        self.mid_pose_list = gui_cfg_params['gui_franka']['tab_russ']['mid_pose']
                        
                        z_backward = np.identity(4)
                        z_backward[2, 3] = -init_upper_length      
                        self.middle_pose = self.ros_motion.get_tfmat_from_trans_quat(self.mid_pose_list[:3], self.mid_pose_list[3:])
                        pos, quat = self.pose_dmp.forward(s=0.0)
                        self.init_upper_pose = self.ros_motion.get_tfmat_from_trans_quat(pos, quat)
                        self.init_upper_pose = np.dot(self.init_upper_pose, z_backward)
                        pos, quat = self.pose_dmp.forward(s=1.0)
                        self.final_pose = self.ros_motion.get_tfmat_from_trans_quat(pos, quat)
                        print('initialization done! start scanning ...')
                        
                        scan_state = 1
                        flag_start_scan = True
                    if not flag_start_interact and rospy.get_param('/gui_franka/russ/make_contact'):   # begin interaction
                        rospy.set_param('/gui_franka/russ/make_contact', False)
                        
                        with open(self.franka_gui_dir + '/cfg/gui_franka.yaml') as file:
                            gui_cfg_params = yaml.load(file, Loader=yaml.FullLoader)
                        
                        # define middel pose during the scan
                        self.home_jnt_config = gui_cfg_params['gui_franka']['tab_russ']['home_jnt_config']
                        self.mid_jnt_config = gui_cfg_params['gui_franka']['tab_russ']['mid_jnt_config']
                        self.mid_pose_list = gui_cfg_params['gui_franka']['tab_russ']['mid_pose']
                        
                        z_backward = np.identity(4)
                        z_backward[2, 3] = -init_upper_length      
                        self.middle_pose = self.ros_motion.get_tfmat_from_trans_quat(self.mid_pose_list[:3], self.mid_pose_list[3:])
                        pos, quat = self.pose_dmp.forward(s=0.0)
                        self.init_upper_pose = self.ros_motion.get_tfmat_from_trans_quat(pos, quat)
                        self.init_upper_pose = np.dot(self.init_upper_pose, z_backward)
                        pos, quat = self.pose_dmp.forward(s=1.0)
                        self.final_pose = self.ros_motion.get_tfmat_from_trans_quat(pos, quat)

                        print('initialization done! start interaction ...')
                        scan_state = 1
                        flag_start_interact = True
                        
                    if flag_start_scan:
                        # The sequence of performing the scan from home pose to final pose of the scan path 
                        # 1) moving from home pose to middle pose, then (joint controller with moveit)3
                        # 2) to the upper pose of initial scan point    (impedance cartesian controller with miniJerk linear motion)
                        # 3) make contact                               (directly with hybrid force motion controller)
                        # 4) do the scan and record the image           (hybrid force motion controller with AL DMP)
                        # 5) upper pose of the final scan point         (impedance cartesian controller with miniJerk linear motion)
                        # 6) home pose                                  (joint controller with moveit)
                        # rospy.get_param() # miniJerk p2p movement
                        
                        # 1) home jnt config to middle config
                        if scan_state == 1:
                            self.ros_motion.set_joint_configs(self.mid_jnt_config)
                            print('moving to middle joint configuration done!')
                            
                            # switch to hybrid force motion controller
                            self.ros_motion.switch_hfm_controller()
                            time.sleep(0.2)
                            self.ros_motion.p2p_motion_init(init_pose  = self.middle_pose, 
                                                            final_pose = self.init_upper_pose,
                                                            avg_vel    = 0.03)     # TODO
                            scan_state = 2
                        
                        # 2) middle pose to init upper pose
                        if scan_state == 2:
                            if self.ros_motion.p2p_motion_genration(CtrlMode.Impedance):
                                print('move to middle pose done!')
                                time.sleep(0.5)
                                
                                trans = t.translation_from_matrix(self.init_upper_pose)
                                quat = t.quaternion_from_matrix(self.init_upper_pose)
                                self.ros_motion.pub_cmd_pose(trans=trans, quat=quat, ctrlmode=CtrlMode.HybridFM)
                            
                                scan_state = 3  
                                
                        # 3) make contact
                        if scan_state == 3:
                            if self.contact_alpha > 0.99 and self.force_err < 0.3:
                                time.sleep(0.3) 
                                self.dmp_s = 0
                                scan_state = 4
                                
                                gap_dist = 0.01 
                                delta_s = gap_dist/self.pose_dmp.get_path_length()
                                
                            
                        # 4) sweep scan
                        if scan_state == 4:
                            if self.dmp_s < 1.0 + dmp_vel/self.loop_hz:
                                if self.dmp_s > 1.0:
                                    self.dmp_s = 1.0
                                
                                pos, quat = self.pose_dmp.forward(s=self.dmp_s)                # generated from dmp, where dmp_s could be changed
                                self.ros_motion.pub_planning_target_pose(list(pos)+list(quat)) # publish to rviz
                                self.ros_motion.pub_cmd_pose(trans=pos, quat=quat, ctrlmode=CtrlMode.HybridFM)  
                                
                                self.dmp_s += dmp_vel/self.loop_hz
                                
                                if not flag_reproduce:
                                    if not flag_record_img_start and self.dmp_s > delta_s: 
                                        flag_record_img_start = True
                                        rospy.set_param('/gui_franka/russ/start_record', True)
                                    if not flag_record_img_stop and self.dmp_s > 1.0 - delta_s/2:  
                                        flag_record_img_stop = True
                                        rospy.set_param('/gui_franka/russ/stop_record', True)
                                
                            else:
                                self.dmp_s = 0
                                self.final_pose = self.ros_motion.get_posemat_from_tf_tree( child_frame='panda_EE', 
                                                                                            parent_frame='panda_link0')
                                
                                z_backward[2, 3] = -final_upper_length
                                self.final_upper_pose = np.dot(self.final_pose, z_backward)
                                
                                self.ros_motion.p2p_motion_init(init_pose  = self.final_pose, 
                                                                final_pose = self.final_upper_pose,
                                                                avg_vel    = 0.02)    
                                scan_state = 5
                                flag_record_img_start = False
                                flag_record_img_stop = False
                        
                        # 5) moving upward
                        if scan_state == 5:
                            if self.ros_motion.p2p_motion_genration(CtrlMode.Impedance):
                                time.sleep(2.0)
                                scan_state = 6
                        
                        # 6) move to home
                        if scan_state == 6:
                            # switch to joint trajectory controller
                            self.ros_motion.switch_joint_controller()
                            time.sleep(1.5)
                            self.ros_motion.set_joint_configs(self.home_jnt_config)
                            print('moving to home joint configuration done!')
                            flag_start_scan = False
    
                    if flag_start_interact:
                        # 1) home jnt config to middle config
                        if scan_state == 1:
                            self.ros_motion.set_joint_configs(self.mid_jnt_config)
                            print('moving to middle joint configuration done!')
                            
                            # switch to hybrid force motion controller
                            self.ros_motion.switch_hfm_controller()
                            time.sleep(0.2)
                            self.ros_motion.p2p_motion_init(init_pose  = self.middle_pose, 
                                                            final_pose = self.init_upper_pose,
                                                            avg_vel    = 0.04)     # TODO
                            scan_state = 2
                        
                        # 2) middle pose to init upper pose
                        if scan_state == 2:
                            if self.ros_motion.p2p_motion_genration(CtrlMode.Impedance):
                                print('move to middle pose done!')
                                time.sleep(0.5)
                                
                                trans = t.translation_from_matrix(self.init_upper_pose)
                                quat = t.quaternion_from_matrix(self.init_upper_pose)
                                self.ros_motion.pub_cmd_pose(trans=trans, quat=quat, ctrlmode=CtrlMode.HybridFM)
                            
                                scan_state = 3 
                                
                        # 3) make contact
                        if scan_state == 3:
                            if self.contact_alpha > 0.99 and self.force_err < 0.3:
                                time.sleep(0.3) 
                                self.dmp_s = 0
                                scan_state = 4
                                print('begin to sweep!')
                        
                        # 4) sweep scan
                        if scan_state == 4:
                            if not flag_move_forward and rospy.get_param('/gui_franka/russ/move_forward'):
                                rospy.set_param('/gui_franka/russ/move_forward', False)
                                flag_move_forward = True
                                
                            if flag_move_forward:
                                if self.dmp_s < 1.0 + dmp_vel/self.loop_hz:
                                    if self.dmp_s > 1.0:
                                        self.dmp_s = 1.0
                                    pos, quat = self.pose_dmp.forward(s=self.dmp_s)                # generated from dmp, where dmp_s could be changed
                                    self.ros_motion.pub_planning_target_pose(list(pos)+list(quat)) # publish to rviz
                                    self.ros_motion.pub_cmd_pose(trans=pos, quat=quat, ctrlmode=CtrlMode.HybridFM)  
                                    
                                    self.dmp_s += dmp_vel/self.loop_hz
                            
                            if not self.footswitch_pressed:  # can modified self.dmp_sso
                                flag_move_forward = False 
                                flag_adjust_force = False
                                
                                # print('foot switcher pressed ')
                                force_ext = -self.intent_detector.force_est[0] - self.intent_detector.force_cali[0]   # Mind the direction of measured forces
                                print('external force: {} N'.format(force_ext))
                                ds = self.intent_detector.path_interact(force_ext=force_ext, total_dist=self.pose_dmp.get_path_length(), 
                                                                        scale_factor=0.07, force_lim=4.5, deadzone=1.0) 
                                self.dmp_s += ds/self.loop_hz
                                if self.dmp_s > 1.0:
                                    self.dmp_s = 1.0
                                elif self.dmp_s < 0.0:
                                    self.dmp_s = 0.0
                                
                                # publish commands
                                # if abs(ds) > 0.0+1e-8:
                                pos, quat = self.pose_dmp.forward(s=self.dmp_s)                # generated from dmp, where dmp_s could be changed
                                self.ros_motion.pub_planning_target_pose(list(pos)+list(quat)) # publish to rviz
                                self.ros_motion.pub_cmd_pose(trans=pos, quat=quat, ctrlmode=CtrlMode.HybridFM)  
                            else:
                                # change the contact forces
                                if not flag_adjust_force:
                                    self.force_adj = self.des_force # do some initial operations 
                                    flag_adjust_force = True
                                else:
                                    # start adjusting contact force
                                    bias = 3.0 
                                    inter_force = self.des_force - (self.intent_detector.force_est[2] + bias)
                                    
                                    print('desired force: ', self.des_force)
                                    print('self.intent_detector.force_est[2]: ', self.intent_detector.force_est[2])
                                    
                                    self.inter_force_limit = 4.0
                                    self.scan_deadzone = 1.5
                                    if abs(inter_force) > self.scan_deadzone:
                                        print(f'inter_force: {inter_force} N')
                                        self.scan_sf = 4.0   # scale factor for mapping the externel force into velocity
                                        if inter_force > self.scan_deadzone: # pushing downward, increase the contact force
                                            self.force_adj += self.intent_detector.inter_force_clamp(inter_force - self.scan_deadzone, 
                                                                                    self.inter_force_limit)*self.scan_sf/self.loop_hz
                                        else:
                                            self.force_adj += self.intent_detector.inter_force_clamp(inter_force + self.scan_deadzone, 
                                                                                    self.inter_force_limit)*1.5*self.scan_sf/self.loop_hz
                                        if self.force_adj > self.intent_detector.scan_max_force:
                                            self.force_adj = self.intent_detector.scan_max_force
                                        elif self.force_adj < self.intent_detector.scan_min_force:
                                            self.force_adj = self.intent_detector.scan_min_force
                                        # print('adjust force: ', self.force_adj, ' N')
                                        self.intent_detector.pub_des_force(self.force_adj)
                                                        
                            
                            if rospy.get_param('/gui_franka/russ/back_home'):
                                self.dmp_s = 0
                                self.final_pose = self.ros_motion.get_posemat_from_tf_tree( child_frame='panda_EE', 
                                                                                            parent_frame='panda_link0')
                                z_backward[2, 3] = -final_upper_length
                                self.final_upper_pose = np.dot(self.final_pose, z_backward)
                                self.ros_motion.p2p_motion_init(init_pose  = self.final_pose,  
                                                                final_pose = self.final_upper_pose,
                                                                avg_vel    = 0.02)           
                                rospy.set_param('/gui_franka/russ/back_home', False)
                                
                                scan_state = 5
                        
                        # 5) moving upward
                        if scan_state == 5:
                            if self.ros_motion.p2p_motion_genration(CtrlMode.Impedance):
                                time.sleep(2.0)
                                scan_state = 6
                        
                        # 6) move to home
                        if scan_state == 6:
                            # switch to joint trajectory controller
                            self.ros_motion.switch_joint_controller()
                            time.sleep(1.0)
                            self.ros_motion.set_joint_configs(self.home_jnt_config)
                            print('moving to home joint configuration done!')
                            flag_start_interact = False
                    
                    
if __name__ == '__main__':
    pcd_savepath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+'/tmp' + '/tmp.pcd'
    print('pcd_sacepath: ', pcd_savepath)
    pos_dmp = AL_PosDMP(n_kfns=41, th=3, k=3.5)
    ort_dmp = AL_OrtDMP(n_kfns=81, th=3, k=3.5)
    pose_dmp = RUSS_DMP(pos_dmp, ort_dmp)
    try:
        franka_path = FrankaPath(pcd_savepath, pose_dmp=pose_dmp, loophz=60)
        franka_path.run()
    except rospy.ROSInterruptException:
        pass
