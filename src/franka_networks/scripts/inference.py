#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye dianye.huang@tum.de
LastEditTime: 2024-01-04 19:10:45
Description: 
    inference of the trained network, and ros interface for 
    online segmentation
'''

import os
from training import get_transforms, get_label_invtransforms
from unet_utils import CreateUltraSoundDataSet

import torch
from torch.utils.data import DataLoader
from training import get_model_dict, get_loss_fn_dict, get_score_fn_dict

from cv2 import cv2
import numpy as np
import time
import PIL.Image as pilimage

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

import rospy
import tf.transformations as t
from sensor_msgs.msg import Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge

class UNetRosInterface(object):
    '''
    Description: 
        UNetRosInterface gets the robot_pose and the US image from 
        
    '''    
    def __init__(self,
                usimg_topic_name,
                eepose_topic_name):
        # infos
        self.usimg = None # np.array
        self.eemat = np.identity(4) # np.array
        
        # ros init
        self.flag_img_update = False
        self.bridgeC = CvBridge()
        self.sub_usimg = rospy.Subscriber(usimg_topic_name, Image, self.sub_usimg_cb, queue_size=1)
        self.sub_eepose = rospy.Subscriber(eepose_topic_name, PoseStamped, self.sub_eepose_cb, queue_size=1)
        print('<UNetRosInterface> Waiting for topic: ' + usimg_topic_name)
        rospy.wait_for_message(usimg_topic_name, Image, timeout=4.0)
        # print('<UNetRosInterface> Waiting for topic: ' + eepose_topic_name)
        # rospy.wait_for_message(eepose_topic_name, PoseStamped, timeout=4.0)
        print('<UNetRosInterface> All required topics received!')        
        img_size = self.usimg.shape
        print('<UNetRosInterface> US image size: {}.'.format(img_size))
        print('<UNetRosInterface> Initialization done!')


    def sub_usimg_cb(self, msg:Image):
        '''
        Description: 
            subscribe ultrasound image from ros, with data type Image under the sensor_msg
            the received image will be converted to numpy.ndarray by virtual of CvBridge
        @ param : msg{Image}
        '''        
        self.flag_img_update = True
        self.usimg = self.bridgeC.imgmsg_to_cv2(msg, desired_encoding="passthrough")
    
    def sub_eepose_cb(self, msg:PoseStamped):
        '''
        Description: 
            subscribe pose of the probe with data type PoseStamped, and converted it 
            into a 4x4 homogeneous matrix
        @ param : msg{PoseStamped}
        '''
        trans = np.array([msg.pose.position.x, 
                        msg.pose.position.y, 
                        msg.pose.position.z])
        quat = [msg.pose.orientation.x, 
                msg.pose.orientation.y, 
                msg.pose.orientation.z, 
                msg.pose.orientation.w]
        self.eemat = t.concatenate_matrices(t.translation_matrix(trans), 
                                            t.quaternion_matrix(quat))    
    
    def get_img_size(self):
        return self.usimg.shape

    def get_usimg(self):
        '''
        Description: 
            the us image size from the framegrabber is (H, W)
        '''        
        flag_update = self.flag_img_update
        self.flag_img_update = False
        return flag_update, self.usimg.copy()
    
    def get_eepose(self):
        return self.eemat.copy()
    
    def get_imgmsg(self, img:np.array):
        return self.bridgeC.cv2_to_imgmsg(img)


def get_image_centroid(seg_img):
    '''
    Description: 
        get the center point of the segmented image
    @ param : seg_img{np.array, float32} -- the tensity of each pixel is in (0, 1) 
    @ param : contour{np.array, int}     -- 
    '''    
    uint_img = np.array(seg_img*255.0).astype('uint8')
    _, bin_img = cv2.threshold(uint_img, 127, 255, 0) # binary image
    contours,_ = cv2.findContours(bin_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    centroids=[]
    for c in contours:
        area=cv2.contourArea(c)
        if area<1000:
            continue
        M = cv2.moments(c)
        try:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        except:
            continue
        centroids.append([cX,cY])
    return np.array(centroids)

from tqdm import tqdm
if __name__ == '__main__':
    # get path of the package directory 
    package_dir = os.path.dirname(os.path.dirname(__file__))
    # loading model 
    model_path = package_dir + '/model/bestmodel.pt'
    unet_model_inference = UNetModelInference(model_path=model_path, 
                                            data_trans=get_transforms)
    
    rospy.init_node('unet_node', anonymous=True)
    pub_segres_visimg = rospy.Publisher('/unet/segmented_vis_img', Image, queue_size=1) # a compounded img for visualization
    pub_segres_img = rospy.Publisher('/unet/segmented_img', Image, queue_size=1)        # a compounded img for visualization
    unet_ros_interface = UNetRosInterface(usimg_topic_name='/frame_grabber/us_img',
                                        eepose_topic_name='/franka_state_controller/cartesian_pose')
    # get label reverse transformation
    img_size = unet_ros_interface.get_img_size()
    label_invtrans = get_label_invtransforms(resize=img_size)
    
    rate = rospy.Rate(50)
    while not rospy.is_shutdown():
        # get usimg from ros
        flag_update, usimg = unet_ros_interface.get_usimg()
        if flag_update:
        # check inference and processing time collapse
            
            # run the model to get segmented result
            seg_img = unet_model_inference.inference(usimg)
            # process the segmented result
                        # start_time = time.time()
            kernel = np.ones((2,2),np.uint8)
            seg_img = cv2.morphologyEx(seg_img, cv2.MORPH_OPEN, kernel)
            
            seg_img = np.array(seg_img*255).astype('uint8')
            _, seg_img = cv2.threshold(seg_img,127,255,0) # converted into binary image
            
            # resize seg_img
            seg_img = label_invtrans(seg_img)
            seg_img = np.array(seg_img).astype('uint8')
            
            # detect the location of vessel
            # detect contour and find the circle
            # https://www.cnblogs.com/wj-1314/p/11510789.html 
            vis_img = cv2.cvtColor(usimg, cv2.COLOR_GRAY2BGR)
            vis_img[np.where(seg_img>127)] = (0, 0, 255)
            contours,_ = cv2.findContours(seg_img, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            centers=[]
            for c in contours:
                area=cv2.contourArea(c)
                if area < 800:
                    continue
                (x, y), radius = cv2.minEnclosingCircle(c)
                centers.append([x, y])
            if len(centers):
                center = (int(x), int(y))
                radius = int(radius)
                vis_img = cv2.circle(vis_img, center, radius, (0, 255, 0), 4)
            
            # publish the result image
            img_msg = unet_ros_interface.get_imgmsg(seg_img)
            pub_segres_img.publish(img_msg)
            
            img_msg = unet_ros_interface.get_imgmsg(vis_img)
            pub_segres_visimg.publish(img_msg)
        
        rate.sleep()
    

