#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-10-16 15:36:41
LastEditors: Dianye Huang
LastEditTime: 2024-01-04 18:23:27
Description: 
    This script loads the trained network, receives the US images from
    the rostopic, segments the vessels, and records the center points
    of the segmented results for the downstream path refinement tasks.
'''

import rospy 
from cv_bridge import CvBridge 
import tf.transformations as t
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped


import os
import cv2
import pickle
import time
from tqdm import tqdm
import PIL.Image as pilimage


import torch
from torch.utils.data import DataLoader
from training import get_transforms, get_label_invtransforms
from training import get_model_dict, get_loss_fn_dict, get_score_fn_dict
from unet_utils import CreateUltraSoundDataSet

class UNetModelInference(object):
    '''
    Description: 
        UNet inference receive batch of test data for evaluation or a single image for inference
        
    @ param : model_path{str}                        -- trained model location  
    @ param : data_trans{torchvision.transforms}     -- image and label transformations for input data
    '''    
    def __init__(self, 
                model_path,
                data_trans,
                device=None):
        print('<UNetModelInference> Loading model ...')
        start_time = time.time()
        self.model, resize, self.loss_fn, self.score_fn = self.get_model_from_ckp(model_path)
        self.img_trans, self.label_trans = data_trans(resize=resize)
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        self.model = self.model.to(self.device)
        time_collapse = round(time.time() - start_time, 2)
        print('<UNetModelInference> Time collapse: {} seconds.'.format(time_collapse))
        print('<UNetModelInference> Model loaded to {}, model initialization done!'.format(self.device))
        
    def get_model_from_ckp(self, checkpoint_fpath):
        '''
        Description: 
            This function is high coupled with the save_ckp() function in unet_utils.py
        @ param : checkpoint_fpath{str} -- checkpoint file path that saves the best model during training
        '''    
        checkpoint = torch.load(checkpoint_fpath)
        model_key = checkpoint['run_params']['network']
        loss_key = checkpoint['run_params']['loss_fn']
        score_key = checkpoint['run_params']['score_fn']
        # load model
        model = get_model_dict()[model_key]
        model.load_state_dict(checkpoint['state_dict'])
        model = model.eval()    # disable batch normalization and dropout, the opposite case would be in train() mode 
                                # https://blog.csdn.net/wuqingshan2010/article/details/106013660
        # load transform and functions
        resize = checkpoint['run_params']['resize']
        loss_fn = get_loss_fn_dict()[loss_key]
        score_fn = get_score_fn_dict()[score_key]
        return model, resize, loss_fn, score_fn

    def model_test(self, data_dir, batch_size=10):
        '''
        Description: 
            test model with data from the specified data directory, whose child directory should 
            includes mask and image
        @ param : data_dir{str}   -- directory
        @ param : batch_size{int} -- batch size for dataloader
        '''        
        print('loading data ...')
        dataset = CreateUltraSoundDataSet(root_dir=data_dir, 
                                        transforms=(self.img_trans, self.label_trans))
        dataset.disp_info()
        test_dataloader = DataLoader(dataset=dataset, batch_size=batch_size)
        # ------------------------- test loop -------------------------
        print('start testing ...')
        total_loss = 0
        total_score = 0
        with torch.no_grad():
            pbar = tqdm(test_dataloader, desc = 'description')
            for batch in pbar:
                # load data
                images = batch[0].to(device=self.device, dtype=torch.float32)
                labels = batch[1].to(device=self.device, dtype=torch.float32)
                # calc loss
                preds = self.model(images)
                seg_loss = self.loss_fn(preds, labels)
                score = self.score_fn(preds, labels)
                total_loss += seg_loss.item()*len(batch[0])
                total_score += score.item()*len(batch[0])
                pbar.set_description(f"batch test loss: {round(seg_loss.item(), 6)}, score: {round(score.item(),6)}")
        print('testing result, loss:', total_loss/len(test_dataloader.dataset), ', score:', total_score/len(test_dataloader.dataset))

    def inference(self, single_img):
        '''
        Description: 
            infer a single image to get a segmented image
        @ param : single_img{np.array}   -- image with the shape of (H, W)
        @ return: seg_img{np.array}      -- segmented image for subsequent processing
        '''    
        single_img = pilimage.fromarray(single_img) # will automatically add a new axis
        with torch.no_grad():
            img_batch = self.img_trans(single_img).unsqueeze(0) # img_batch.size() = (1, 1, resize, resize)
            img_batch = img_batch.to(self.device, dtype=torch.float32)
            seg_img = self.model(img_batch)
            seg_img = seg_img.squeeze(0)
            seg_img = seg_img.squeeze(0)
        return seg_img.cpu().numpy()


import numpy as np
def get_pixel_2_point_tf(im_h, im_w, trans_w=0.0375, trans_d=0.055, epsilon=0.0):
    Lp = trans_w  # length/width of transducer at the tip of the probe
    Di = trans_d  # depth of image
    return np.array([
        [0,         0, -1,        0],
        [0,   Lp/im_w,  0,    -Lp/2],
        [Di/im_h  , 0,  0,  epsilon],
        [0,         0,  0,        1]])

def pixel_2_world(pixel_xy:list, ee_pose:np.array, probeTimg:np.array):
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

class NetRecordImgPose:
    def __init__(self, savedir, model_path):
        rospy.init_node('save_usimg_eepose_node', anonymous=True)
        
        self.savedir = savedir
        self.cvbridge = CvBridge()
        self.flag_img_update = False
        self.eepose = None
        self.save_dict = None
        self.usimg = None
        
        rospy.set_param('/gui_franka/russ/start_record', False)
        rospy.set_param('/gui_franka/russ/stop_record', False)
        
        self.flag_start_record = False
        
        rospy.Subscriber(   '/frame_grabber/us_img', Image, 
                            self.usimg_cb, queue_size=1)
        rospy.Subscriber(   '/franka_state_controller/cartesian_pose', PoseStamped,
                            self.eepose_cb, queue_size=1)

        rospy.wait_for_message('/frame_grabber/us_img', Image, timeout=10.0)
        rospy.wait_for_message('/franka_state_controller/cartesian_pose', PoseStamped, timeout=10.0)

        img_size = self.usimg.shape
        self.unet_model_inference = UNetModelInference( model_path=model_path, 
                                                        data_trans=get_transforms)
        self.label_invtrans = get_label_invtransforms(resize = img_size)
        self.probeTimg = get_pixel_2_point_tf(img_size[0], img_size[1], trans_w=0.0375, trans_d=0.055, epsilon=0.0)  # TODO

        save_idx = 0
        rate = rospy.Rate(10)   # TODO to be configured

        while not rospy.is_shutdown():
            rate.sleep()
            if self.flag_img_update: # and wait for saving signal
                self.flag_img_update = False
                if not self.flag_start_record:
                    if rospy.get_param('/gui_franka/russ/start_record'):
                        rospy.set_param('/gui_franka/russ/start_record', False)
                        self.flag_start_record = True
                        self.save_dict = dict()
                        self.save_dict['us_images'] = list()
                        self.save_dict['ee_poses'] = list()
                        self.center_pos_dict = dict()
                        save_idx = 0
                else:
                    center_pts = self.get_center_pts_from_unet(self.usimg, self.eepose, self.probeTimg)
                    self.center_pos_dict[str(save_idx)] = center_pts
                    
                    self.save_dict['us_images'].append(self.usimg)
                    self.save_dict['ee_poses'].append(self.eepose)
                    print('save idx: ', save_idx)
                    save_idx += 1
                    if rospy.get_param('/gui_franka/russ/stop_record'):
                        rospy.set_param('/gui_franka/russ/stop_record', False)
                        self.flag_start_record = False
                        cv2.destroyAllWindows()
                        # save dict
                        self.save_dict['center_pos'] = self.center_pos_dict
                        self.save_dict['num_data'] = save_idx
                        # filename = 'usimg_eepose_' + str(rospy.Time.now()) + '.pickle' # TODO uncomment if save demonstration 
                        filename = 'usimg_eepose_tmp.pickle' 
                        self.save_pickle(self.save_dict, self.savedir + filename)
                        print('length of record data: ', len(self.save_dict['ee_poses']))
    
    def get_center_pts_from_unet(self, usimg, ee_pose, probeTimg):
        seg_img = self.unet_model_inference.inference(usimg)  # TODO UNET for segmentation
        
        kernel = np.ones((2,2),np.uint8)
        seg_img = cv2.morphologyEx(seg_img, cv2.MORPH_OPEN, kernel)
        
        seg_img = np.array(seg_img*255).astype('uint8')
        _, seg_img = cv2.threshold(seg_img,127,255,0) # converted into binary image
        
        # resize seg_img
        seg_img = self.label_invtrans(seg_img)
        seg_img = np.array(seg_img).astype('uint8')
        
        # get point clouds
        baseTprob = t.quaternion_matrix(ee_pose[3:])
        baseTprob[:3, 3] = np.array(ee_pose[:3])
        
        
        # detect the location of vessel
        # detect contour and find the circle
        # might the different image coordinate system in cv::mat and numpy.array when presenting an image
        # https://www.cnblogs.com/wj-1314/p/11510789.html 
        vis_img = cv2.cvtColor(usimg, cv2.COLOR_GRAY2BGR)
        vis_img[np.where(seg_img>127)] = (0, 0, 255)
        contours,_ = cv2.findContours(seg_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        
        # print('num of contours: ', len(contours))
        centers=[]
        for c in contours:
            area=cv2.contourArea(c)
            if area < 200:
                continue
            (x, y), radius = cv2.minEnclosingCircle(c)
            vis_img = cv2.circle(vis_img, (int(x), int(y)), int(radius), (0, 255, 0), 4)
            # centers.append([x, y])
            centers.append([y, x])
        
        cv2.imshow('seg_res_img', vis_img)
        cv2.waitKey(1)
        
        if not len(centers):
            return None
        
        center_pts = pixel_2_world(centers, baseTprob, probeTimg)
        return center_pts
    
    def save_pickle(self, data_dict, savepath):
        with open(savepath, 'wb') as f:
            pickle.dump(data_dict, f)
    
    def usimg_cb(self, msg:Image):
        self.usimg = self.cvbridge.imgmsg_to_cv2(msg)
        self.flag_img_update = True
    
    def eepose_cb(self, msg:PoseStamped):
        self.eepose = [ msg.pose.position.x,
                        msg.pose.position.y,
                        msg.pose.position.z,
                        msg.pose.orientation.x,
                        msg.pose.orientation.y,
                        msg.pose.orientation.z,
                        msg.pose.orientation.w ]

if __name__ == '__main__':
    savedir = '/home/hdy/Projects_ws/RobotAssistedDVT_tii/franka_russ_ws/src/path_plan/franka_networks/data/'
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = package_dir + '/model/bestmodel.pt'
    nrip = NetRecordImgPose(savedir, model_path)
