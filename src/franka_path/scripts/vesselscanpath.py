#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-10-16 15:36:41
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-01-04 19:09:52
Description: 
    This script 
'''

import pickle
import open3d as o3d
import numpy as np

import cv2
import tf.transformations as t

# import matplotlib.pyplot as plt
from dmp_utils import AL_PosDMP, AL_OrtDMP
from posort_dmp import RUSS_DMP

import rospy
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Pose, PoseArray

from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField

class GetImagePixel:
    def __init__(self, windowname="Get pixel coordinate from image"):
        self.pixel_xy = None
        self.windowname = windowname
        
    def get_pixel_from_img(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        cv2.namedWindow(self.windowname)
        cv2.setMouseCallback(self.windowname, self.on_EVENT_LBUTTONDOWN, img)
        cv2.imshow(self.windowname, img)
        while True:
            k = cv2.waitKey(10)
            if k  == ord('q') or k == ord('Q'):
                break
            if k == ord('a') or k == ord('A'):
                return -1, None
            if k == ord('d') or k == ord('D'):
                return 1, None
        return 0, self.pixel_xy

    def on_EVENT_LBUTTONDOWN(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            xy = "%d,%d" % (x, y)
            cv2.circle(param, (x, y), 5, (0, 0, 255), thickness=-1)
            cv2.putText(param, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(self.windowname, param)
            self.pixel_xy = [y, x]

class VesselScanPath:
    def __init__(   self, 
                    posearr_topic='path_planning/skin/final_scan_pose', 
                    marker_topic='path_planning/skin/final_markers', 
                    center_topic='path_planning/skin/center_line',
                    vessel_topic='path_planning/skin/vessel_line',
                    ros_node=False):
        if ros_node:
            rospy.init_node('vesselscanpath_node', anonymous=True)
        self.pub_posearray = rospy.Publisher(posearr_topic, PoseArray, queue_size=1)
        self.pub_markers = rospy.Publisher(marker_topic, Marker, queue_size=1)
        self.pub_centers = rospy.Publisher(center_topic, Marker, queue_size=1)
        self.pub_vessels = rospy.Publisher(vessel_topic, Marker, queue_size=1)
        
        self.scan_pose = None
        self.centerline_arr = None
        self.centerlines = None
    
    def load_pickle(self, filepath):
        with open(filepath, 'rb') as f:
            res = pickle.load(f)
        print(res.keys())
        return res
    
    def load_pc_from_pcd(self, filepath):
        pcd_res = o3d.io.read_point_cloud(filepath)
        return np.array(pcd_res.points)

    def get_pixel_2_point_tf(self, im_h, im_w, trans_w=0.0375, trans_d=0.055, epsilon=0.0):
        Lp = trans_w  # length/width of transducer at the tip of the probe
        Di = trans_d  # depth of image
        return np.array([
            [0,         0, -1,        0],
            [0,   Lp/im_w,  0,    -Lp/2],
            [Di/im_h  , 0,  0,  epsilon],
            [0,         0,  0,        1]])

    def pixel_2_world(self, pixel_xy:list, ee_pose:np.array, probeTimg:np.array):
        '''
        pixel_xy : N x 2
        ee_pose  : 4 x 4
        probeTimg: 4 x 4
        points : 4 x N
        '''
        num_pixels = len(pixel_xy)
        pixels_arr = np.array(pixel_xy).reshape(-1, 2)
        pixels_arr = np.concatenate((pixels_arr, np.zeros((num_pixels, 1)), 
                                        np.ones((num_pixels, 1))), axis=1)
        pixels_arr = pixels_arr.transpose()
        baseTimg = np.dot(ee_pose, probeTimg).reshape(1, 4, 4)
        points = np.dot(baseTimg, pixels_arr).reshape(4, -1)
        
        return points.transpose()

    def choose_vessel(self, us_images, poses):
        img_size = np.array(us_images[0]).shape
        probeTimg = self.get_pixel_2_point_tf(img_size[0], img_size[1], trans_w=0.0375, trans_d=0.055, epsilon=0.0)
        pixel_coor = GetImagePixel()
        start_img_idx = 0
        max_img_idx = len(us_images)
        while True:
            usimg = us_images[start_img_idx]
            ret, pixel_xy = pixel_coor.get_pixel_from_img(usimg)
            if ret == 0:
                print('chosen coordinate: ', pixel_xy)
                break
            if ret == -1:
                start_img_idx -= 1
                start_img_idx = max(start_img_idx, 0)
            if ret == 1:
                start_img_idx += 1
                start_img_idx = min(start_img_idx, max_img_idx-1)
        cv2.destroyAllWindows()
        startTprob = t.quaternion_matrix(poses[start_img_idx][3:])
        startTprob[:3, 3] = np.array(poses[start_img_idx][:3])
        start_pos = self.pixel_2_world([pixel_xy], startTprob, probeTimg)
        return start_img_idx, start_pos[0, :3]

    def get_centerline_position(self, start_pos, start_idx, pos_dict, data_length, flag_fit=True):
        last_pos = start_pos
        centerline_list = list()
        self.centerlines = None
        for idx in range(data_length):
            if idx < start_idx:
                continue
            center_pos = pos_dict[str(idx)][:, :3]
            
            if self.centerlines is None:
                self.centerlines = center_pos.copy()
            else:
                self.centerlines = np.concatenate((self.centerlines, center_pos), axis=0)
                
            if center_pos is not None:
                dist_arr = np.sqrt(np.sum(np.power(center_pos-last_pos, 2), axis=1))
                min_idx = np.argmin(dist_arr)
                last_pos = center_pos[min_idx]
                centerline_list.append(last_pos[:3])
        if flag_fit:
            # dmp fitting
            pos_dmp = AL_PosDMP(n_kfns=31, th=2, k=2)
            total_dist, s_arr, pos_arr = pos_dmp.fit(   sampleHz=30, position_list=centerline_list, 
                                                        flag_visresult=False, vel_lim=0.01)
            # plt.show()
            centerline_list = pos_dmp.rollout(s_arr)
        return np.array(centerline_list)

    def project_path_to_skin(self, pc_arr, centerline_arr):
        probe_axis = list()
        final_scan_path = list()
        pc_arr_xy = pc_arr[:, :2]
        for p in centerline_arr:
            dist = np.sqrt(np.sum(np.power(pc_arr_xy-p[:2], 2), axis=1))
            scan_point = pc_arr[np.argmin(dist), :3]
            final_scan_path.append(scan_point)
            probe_axis.append((p-scan_point)/np.linalg.norm(p-scan_point))
        probe_axis = np.array(probe_axis)
        return np.array(final_scan_path), probe_axis

    def get_scan_pose(self, pc_arr, centerline_arr):
        final_path, probe_axis = self.project_path_to_skin(pc_arr, centerline_arr)
        ort_dmp = AL_OrtDMP(n_kfns=41, th=2, k=2)
        pos_dmp = AL_PosDMP(n_kfns=31, th=2, k=2)
        pose_dmp = RUSS_DMP(pos_dmp, ort_dmp)
        s_arr = pose_dmp.fit(final_path, probe_axis, sampleHz=10, flag_vis_result=False, pose_method='3')
        pos_arr, quat_arr = pose_dmp.rollout(s_arr)
        scan_pose =  np.hstack((pos_arr, quat_arr))
        return pose_dmp, scan_pose

    def get_scan_path(self, pickle_path, pcd_path):
        # load data
        pickle_ret = self.load_pickle(pickle_path)
        pc_arr = self.load_pc_from_pcd(pcd_path)
        us_images = pickle_ret['us_images']
        poses = pickle_ret['ee_poses']
        data_length = pickle_ret['num_data']
        pos_dict = pickle_ret['center_pos']
        # choose vessel to be scanned
        start_img_idx, start_pos = self.choose_vessel(us_images, poses)
        # get centerline position
        centerline_arr = self.get_centerline_position(start_pos, start_img_idx, pos_dict, data_length)
        self.centerline_arr = centerline_arr
        # get scan pose
        pose_dmp, scan_pose = self.get_scan_pose(pc_arr, centerline_arr)    
        self.scan_pose = scan_pose
        # print('scan pose: ', scan_pose.shape)
        return pose_dmp, scan_pose
    
    def generate_msg_Marker(self, traj_data, frame_str, ns_str, rgba_list):
        '''
        traj_data: path for visualization
        frame_str: base frame id
        ns_str: namespace
        rbga_list: to color the path for visualization
        '''
        points = Marker()
        points.header.frame_id = frame_str
        points.header.stamp = rospy.Time.now()

        points.ns = ns_str
        points.action = Marker.ADD
        points.pose.orientation.w = 1.0 # no orientation

        points.id = 0
        points.type = Marker.POINTS

        points.scale.x = 0.002   # define scales
        points.scale.y = 0.002
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

    def get_posearray_msg(self, pose_list, frame_id='panda_link0'):
        pose_array = PoseArray()
        pose_array.header.frame_id = frame_id
        for p in pose_list:
            pose = Pose()
            pose.position.x = p[0]
            pose.position.y = p[1]
            pose.position.z = p[2]
            pose.orientation.x = p[3]
            pose.orientation.y = p[4]
            pose.orientation.z = p[5]
            pose.orientation.w = p[6]
            pose_array.poses.append(pose)
        pose_array.header.stamp = rospy.Time.now()
        return pose_array

    def get_point_cloud(self, pc_list, color_list=None, pc_frame_id = "panda_link0"):
        if color_list is None:
            pc_array = np.array(pc_list)
            dtype = np.float32
            itemsize = np.dtype(dtype).itemsize
            data = pc_array.astype(dtype).tobytes()
            fields = [
                PointField('x', 0,  PointField.FLOAT32, 1),
                PointField('y', 4,  PointField.FLOAT32, 1),
                PointField('z', 8,  PointField.FLOAT32, 1),
                PointField('intensity', 12, PointField.FLOAT32, 1)
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

    def publish_final_posearr(self):
        posearr_msg = self.get_posearray_msg(self.scan_pose)
        self.pub_posearray.publish(posearr_msg)
    
    def publish_final_markers(self):
        marker_msg = self.generate_msg_Marker(  self.scan_pose[:, :3], frame_str='panda_link0', 
                                                ns_str='final_scan_path', rgba_list=[0.0, 1.0, 1.0, 1.0])
        self.pub_markers.publish(marker_msg)
    
    def publish_final_centerlines(self):
        marker_msg = self.generate_msg_Marker(  self.centerlines[:, :3], frame_str='panda_link0', 
                                                ns_str='vessel_centerline', rgba_list=[1.0, 0.0, 0.0, 1.0])
        self.pub_centers.publish(marker_msg)
    
    def publish_final_vessel(self):
        marker_msg = self.generate_msg_Marker(  self.centerline_arr[:, :3], frame_str='panda_link0', 
                                                ns_str='vessel_centerline', rgba_list=[0.0, 0.0, 1.0, 1.0])
        self.pub_vessels.publish(marker_msg)

import os
if __name__ == '__main__':
    # savedir = '/home/hdy/franka_russ_ws/src/path_plan/franka_networks/data/'
    # package_dir = os.path.dirname(os.path.dirname(__file__))
    # model_path = package_dir + '/model/bestmodel.pt'
    # nrip = NetRecordImgPose(savedir, model_path)
    vsp = VesselScanPath()
    filepath = os.path.dirname(os.path.dirname(__file__))+'/data' + '/usimg_eepose_tmp.pickle'
    pcd_file = '/home/hdy/franka_russ_ws/src/path_plan/franka_path/tmp/tmp_bak.pcd'
    vsp.get_scan_path(filepath, pcd_file)
    
