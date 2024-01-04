#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye Huang
LastEditTime: 2022-10-20 16:45:42
Description: 
'''

import torch

def DiceLoss(pred, target, smooth=1.0):
    '''
    Description: 
        dice loss
    @ param : pred{torch.tensor}    -- batch of predictive images
    @ param : target{torch.tensor}  -- batch of label images
    @ return: {torch.tensor}        -- loss
    '''    
    # target = target/255.0  #  if target image is not transformed to tensor , comment if target is 1
    index = (2*torch.sum(pred*target)+smooth)/(torch.sum(pred)+torch.sum(target)+smooth)
    return 1-index

def IoU(pred, target, smooth=1.0):
    '''
    Description: 
        intersection of union
    '''    
    # target = target/255.0 #  if target image is not transformed to tensor, comment if target is 1
    intersection = (pred*target).sum()
    union = (pred + target).sum() - intersection
    IoU = (intersection + smooth)/(union + smooth)
    return IoU


if __name__ == '__main__':
    print('This modules provides some loss or accuracy metric function for the training and testing')
