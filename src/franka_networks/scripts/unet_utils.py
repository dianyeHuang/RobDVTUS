#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye Huang
LastEditTime: 2023-09-22 09:44:11
Description: 
    This module provides interfaces for the network training, e.g.
    - creates us dataset for segmentation purpose
    - running and training management
    - loading and saving models
    Refer to:
    - deeplizard, run builder, and run manager
    - kaggle, save_ckp, load_ckp
'''
import torch
from torch.utils.data import Dataset

import os
import numpy as np
from PIL import Image

import shutil

from collections import OrderedDict
from collections import namedtuple
from itertools import product

import torchvision
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import time
import json

import cv2

# region create_dataset
class CreateUltraSoundDataSet(Dataset):
    def __init__(   self, root_dir, transforms, 
                    train_prefix=None, label_prefix=None, 
                    check_mask=False, vis_labled_img=False):
        self.image_pathlist = list()
        self.label_pathlist = list()
        for id in os.listdir(root_dir+'/mask'):
            if train_prefix is not None:
                id = id[-7:]
            
            imagepath = os.path.join(root_dir+'/image/',train_prefix+id)
            labelpath = os.path.join(root_dir+'/mask/',label_prefix+id)
            
            if check_mask:
                label_map = np.array(Image.open(labelpath))
                label_sum = np.sum(np.sum(label_map))
                
                if label_sum > 0:
                    if vis_labled_img:
                        us_image = np.array(Image.open(imagepath))
                        vis_img = cv2.cvtColor(us_image, cv2.COLOR_GRAY2BGR)
                        vis_img[np.where(label_map>0)] = (0, 0, 255)
                        cv2.imshow('labeled image', vis_img)
                        cv2.waitKey(200)
                    
                    self.image_pathlist.append(imagepath)
                    self.label_pathlist.append(labelpath)
        
        self.transform_image, self.transform_label = transforms
        
    def __len__(self):
        return len(self.image_pathlist)
    
    def __getitem__(self,idx):
        image = Image.open(self.image_pathlist[idx])
        label = Image.open(self.label_pathlist[idx])

        if self.transform_image is not None:
            image = self.transform_image(image)
        
        if self.transform_label is not None:
            label = self.transform_label(label)
        label = torch.Tensor(np.array(label)).unsqueeze(0)
        return image, label
    
    def disp_info(self):
        print('---------- Dataset Info. ---------')
        print('Length of the dataset is:', len(self.image_pathlist))
        image = Image.open(self.image_pathlist[0])
        label = Image.open(self.label_pathlist[0])
        # print('type of image:', type(image))
        print('Original size of the image:', image.size)
        print('Original size of the label:', label.size)
# endregion create_dataset

# region loading_and_saving_model
# refer to: https://www.kaggle.com/code/godeep48/simple-unet-pytorch
def save_ckp(state, is_best, checkpoint_path, best_model_path):
    """
    state: checkpoint we want to save
    is_best: is this the best checkpoint; min validation loss
    checkpoint_path: path to save checkpoint
    best_model_path: path to save best model
    """
    f_path = checkpoint_path
    torch.save(state, f_path)
    if is_best:
        best_fpath = best_model_path
        # copy that checkpoint file to best path given, best_model_path
        shutil.copyfile(f_path, best_fpath)

def load_ckp(checkpoint_fpath, model, optimizer):
    """
    checkpoint_path: path to save checkpoint
    model: model that we want to load checkpoint parameters into       
    optimizer: optimizer we defined in previous training
    """
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    valid_loss_min = checkpoint['valid_loss_min']
    run_params = checkpoint[run_params]
    return model, optimizer, checkpoint['epoch'], valid_loss_min.item()
# endregion loading_and_saving_model

# region runner_and_trainer
class RunBuilder():
    '''
    Description: 
        generate run parameters by producting the possible configurations,
        each run corresponds to a params configuratio of the network
    '''    
    @staticmethod
    def get_runs(params):
        '''
        @ param : params{dict} -- a dictionary that contains the possible configuration for training
        @ return: runs{list}   -- a list whose element is an iterable nametuple
        '''        
        Run = namedtuple('Run', params.keys())
        runs = []
        for v in product(*params.values()):
            runs.append(Run(*v))
        return runs

class struct_Epoch(object):
    '''
    Description: 
        a structure that stores info of an epoch
    '''    
    def __init__(self) -> None:
        self.reset()
    
    def reset(self):
        self.count = 0    
        self.train_totalloss = 0        
        self.valid_totalloss = 0
        self.train_totalscore = 0
        self.valid_totalscore = 0
        self.start_time = None

class struct_Run(object):
    '''
    Description: 
        a structure that stores info of running a network
    '''    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.params = None  
        self.count = 0
        self.start_time = None
        self.data = list()

class RunManager():
    '''
    Description: 
        Runmanager manages the training flow and record the results in an json file 
        as well as tensorboard
    @ param : run{RunBuilder}     -- 
    @ param : network{nn.Modules} -- neutal network model
    @ param : loader{DataLoader}  -- dataloader from torch  
        - workflow:
            for run in RunBuilder.get_runs(param_dict):
                RunManager.begin_run(xxx)
                for epoch in run.num_epoch:
                    RunMangaer.begin_epoch()
                    for batch in loader:
                        xxxx
                        RunManger.track_xxx
                    RunMangaer.end_epoch()
                RunManager.end_run(xxx)
            RunManger.save(filepath)
    '''    
    def __init__(self):
        self.epoch = struct_Epoch()
        self.run = struct_Run()
        self.network = None
        self.loader = None
        self.tb = None

    def begin_run(self, run, network, loader):
        self.run.start_time = time.time()
        self.run.params = run
        self.run.count += 1
        self.network = network
        self.loader = loader
        self.tb = SummaryWriter(comment=f'-{run}')
        # visualize a small batch of network
        images, labels = next(iter(self.loader))
        grid = torchvision.utils.make_grid(images)
        self.tb.add_image('images', grid)
        self.tb.add_graph(self.network
            ,images.to(getattr(run, 'device', 'cpu'))
        )

    def end_run(self):
        self.epoch.reset()
        self.tb.close()

    def begin_epoch(self):
        self.epoch.start_time = time.time()
        self.epoch.count += 1
        self.epoch.train_totalloss = 0
        self.epoch.valid_totalloss = 0
        # self.epoch.total_score = 0
        self.epoch.train_totalscore = 0
        self.epoch.valid_totalscore = 0

    def end_epoch(self, num_train, num_vals, log_params = False):
        # neural networks's result
        epoch_duration = time.time() - self.epoch.start_time
        run_duration = time.time() - self.run.start_time
        train_loss = self.epoch.train_totalloss / num_train   # average loss
        valid_loss = self.epoch.valid_totalloss / num_vals 
        train_score = self.epoch.train_totalscore / num_train
        valid_score = self.epoch.valid_totalscore / num_vals

        # record into the tensorboard
        self.tb.add_scalar('Train_loss', train_loss, self.epoch.count)
        self.tb.add_scalar('Valid_loss', valid_loss, self.epoch.count)
        self.tb.add_scalar('Train_score', train_score, self.epoch.count)
        self.tb.add_scalar('Valid_score', valid_score, self.epoch.count)
        if log_params:
            for name, param in self.network.named_parameters():
                self.tb.add_histogram(name, param, self.epoch.count)
                self.tb.add_histogram(f'{name}.grad', param.grad, self.epoch.count)

        # store an epoch's result
        results = OrderedDict()
        results["run"] = self.run.count
        results["epoch"] = self.epoch.count
        results['train_loss'] = train_loss
        results['train_score'] = train_score
        results['valid_loss'] = valid_loss
        results['valid_score'] = valid_score
        results['epoch duration'] = epoch_duration
        results['run duration'] = run_duration
        for k, v in self.run.params._asdict().items(): 
            results[k] = v
        self.run.data.append(results)

        # display an epoch's result
        df = pd.DataFrame.from_dict(self.run.data, orient='columns')
        print('------------------------ Traning result ----------------------------------------')
        print(df)
        print('--------------------------------------------------------------------------------')

    def track_train_loss(self, loss, batch):
        self.epoch.train_totalloss += loss.item() * batch[0].shape[0] 
    
    def track_valid_loss(self, loss, batch):
        self.epoch.valid_totalloss += loss.item() * batch[0].shape[0] 
    
    def track_train_score(self, score, batch):
        self.epoch.train_totalscore += score.item() * batch[0].shape[0] 
    
    def track_valid_score(self, score, batch):
        self.epoch.valid_totalscore += score.item() * batch[0].shape[0]

    def save_results(self, filepath):
        pd.DataFrame.from_dict(
            self.run.data, orient='columns'
        ).to_csv(f'{filepath}.csv')
        with open(f'{filepath}.json', 'w', encoding='utf-8') as f:
            json.dump(self.run.data, f, ensure_ascii=False, indent=4)
# endregion runner_and_trainer


if __name__ == '__main__':
    print('this module provides some tools for the training of segmentation network')
    
    # root_dir = '/home/hdy/franka_russ_ws/src/path_plan/franka_networks/data/train/1'
    # image_pathlist = list()
    # label_pathlist = list()
    # train_prefix = 'us_train_images'
    # label_prefix = 'Images'
    # for id in os.listdir(root_dir+'/image'):
    #     if train_prefix is not None:
    #         id = id[-8:]
    #     image_pathlist.append(os.path.join(root_dir+'/image/',train_prefix+id))
    #     label_pathlist.append(os.path.join(root_dir+'/mask/',label_prefix+id))
    # print(len(image_pathlist))
    # print(len(label_pathlist))
