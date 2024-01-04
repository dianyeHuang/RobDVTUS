#!/usr/bin/env python
'''
Author: Dianye Huang
Date: 2022-08-18 11:39:53
LastEditors: Dianye Huang
LastEditTime: 2024-01-04 18:25:55
Description: 
    Training management
'''
from unet_utils import CreateUltraSoundDataSet

from torch.utils.tensorboard import SummaryWriter

import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import random_split
from torch.utils.data import DataLoader

from collections import OrderedDict
from unet_utils import RunManager, RunBuilder, save_ckp
from unet.unet3 import UNet3
from loss_zoo import DiceLoss, IoU

from tqdm import tqdm
import numpy as np


def get_transforms(resize=None):
    if resize is not None:
        transform_image = transforms.Compose([
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ])
        transform_label = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.NEAREST),
            #transforms.ToTensor()
        ])
    else:
        transform_image = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(0.5,0.5)
        ])
        transform_label = transforms.Compose([
            #transforms.ToTensor()
        ])
    return (transform_image, transform_label)

def get_label_invtransforms(resize=None):
    if resize is not None:
        invtransform_label = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(resize,interpolation=transforms.InterpolationMode.NEAREST)
        ])
    else:
        invtransform_label = transforms.Compose([
            transforms.ToPILImage()
        ])
    return invtransform_label

def get_model_dict():
    unet3 = UNet3()
    networks = {   # models to be trained
        'unet': unet3
    }
    return networks

def get_loss_fn_dict():
    loss_fns = {   # loss function
        'DiceLoss': DiceLoss
    }
    return loss_fns

def get_score_fn_dict():
    score_fns = {  # accuracy metric
        'IoU': IoU
    }
    return score_fns

# TODO loading check point and continue training is to be implemented
if __name__ == '__main__':  
    # set random seed for reproducibility
    torch.manual_seed(50)
    # np.set_printoptions(precision=5)
    # torch.set_printoptions(precision=4)

    # path to get the label data
    package_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # data_dir = package_dir + '/data/train/1'
    data_dir = package_dir + '/data/train/2'
    save_filepath = package_dir + '/model/trainresult/training_results'
    # --- check point settings
    save_checkpt = True # to get the best model
    valid_loss_min = np.Inf  
    checkpoint_path = package_dir + '/model/checkpoint/chkpoint_'
    best_model_path = package_dir + '/model/bestmodel.pt'
    
    # train_prefix = 'us_train_images'
    train_prefix = 'Images'
    label_prefix = 'Images 1'
    
    # training settings
    networks = get_model_dict()
    loss_fns = get_loss_fn_dict()
    score_fns = get_score_fn_dict()
    params = OrderedDict(
        batch_size = [10],
        num_workers = [2],
        device = ['cuda'],
        network = list(networks.keys()),
        resize = [(256, 256)],
        val_ratio = [0.2],
        num_epoch = [30],
        loss_fn = list(loss_fns.keys()),
        score_fn = list(score_fns.keys())
    )
    
    # start training
    m = RunManager()
    for run in RunBuilder.get_runs(params):
        # load dataset
        transform_image, transform_label = get_transforms(resize=run.resize)
        # dataset = CreateUltraSoundDataSet(root_dir=data_dir, 
        #                                 transforms=(transform_image, transform_label))
        dataset = CreateUltraSoundDataSet(  root_dir=data_dir, 
                                        transforms=(transform_image, transform_label),
                                        train_prefix=train_prefix,
                                        label_prefix=label_prefix,
                                        check_mask=True,
                                        vis_labled_img=False)
        
        num_train = int(len(dataset)*(1 - run.val_ratio))
        num_val = int(len(dataset) - num_train)
        train_set, valid_set = random_split(dataset, [num_train, num_val])
        dataset.disp_info()
        
        # set dataloader
        train_loader = DataLoader(train_set, batch_size=run.batch_size,
                                shuffle=True, num_workers=run.num_workers)
        valid_loader = DataLoader(valid_set, batch_size=run.batch_size,
                                shuffle=True, num_workers=run.num_workers)
        
        # set optimizer
        device = torch.device(run.device)
        model = networks[run.network].to(device)
        model.train() # enable batch normalization and dropout
        loss_function = loss_fns[run.loss_fn]
        score_function = score_fns[run.score_fn]
        optimizer = torch.optim.Adam(model.parameters())
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10)

        m.begin_run(run, model, train_loader)
        print('run settings:', run)
        for epoch in range(run.num_epoch):
            m.begin_epoch()
            
            # ------------------------- training loop -------------------------
            pbar = tqdm(train_loader, desc = 'description')
            for batch in pbar:
                # load data
                images = batch[0].to(device=device, dtype=torch.float32)
                labels = batch[1].to(device=device, dtype=torch.float32)
                # calc loss
                preds = model(images)
                seg_loss = loss_function(preds, labels)
                score = score_function(preds, labels)
                # update weights
                optimizer.zero_grad()
                seg_loss.backward()
                optimizer.step()
                # log data
                m.track_train_loss(seg_loss, batch)
                m.track_train_score(score, batch)
                # display training loss
                pbar.set_description(f"Epoch: {epoch+1}, batch train {run.loss_fn}: {round(seg_loss.item(), 6)}, {run.score_fn}: {round(score.item(),6)}")

            # ------------------------- validataion loop -------------------------
            with torch.no_grad():
                pbar = tqdm(valid_loader, desc = 'description')
                for batch in pbar:
                    # load data
                    images = batch[0].to(device=device, dtype=torch.float32)
                    labels = batch[1].to(device=device, dtype=torch.float32)
                    # calc loss
                    preds = model(images)
                    seg_loss = loss_function(preds, labels)
                    score = score_function(preds, labels)
                    # log data
                    m.track_valid_loss(seg_loss, batch)
                    m.track_valid_score(score, batch)
                    # display validation loss
                    pbar.set_description(f"Epoch: {epoch+1}, batch valid {run.loss_fn}: {round(seg_loss.item(), 6)}, {run.score_fn}: {round(score.item(), 6)}")
            m.end_epoch(num_train=len(train_loader.dataset), num_vals=len(valid_loader.dataset))

            if save_checkpt:
                print('------------ saving checkpoint -----------')
                valid_loss = m.epoch.valid_totalloss/len(valid_loader.dataset)
                checkpoint = {
                    'epoch': epoch + 1,
                    'valid_loss_min': valid_loss,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'run_params': run._asdict()
                }
                save_ckp(checkpoint, False, checkpoint_path, best_model_path)  # TODO how to resume running in this structure?
                if valid_loss <= valid_loss_min:
                    print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min, valid_loss))
                    save_ckp(checkpoint, True, checkpoint_path, best_model_path)
                    valid_loss_min = valid_loss
                print('------------------------------------------')
        m.end_run()
    m.save_results(save_filepath)
    print('traning done!')
