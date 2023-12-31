'''
@copyright ziqi-jin
'''
import argparse
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from datasets import get_dataset
from losses import get_losses
from extend_sam import get_model, get_optimizer, get_scheduler, get_opt_pamams, get_runner

import dill
import os
import glob
import monai
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import SimpleITK as sitk
from statistics import mean
from torch.optim import Adam
import matplotlib.pyplot as plt
from transformers import SamModel
import matplotlib.patches as patches
from transformers import SamProcessor
from IPython.display import clear_output
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import threshold, normalize
import cv2
import dill

from monai.transforms import (
    EnsureChannelFirstd,
    EnsureTyped,
    Compose,
    CropForegroundd,
    CopyItemsd,
    LoadImaged,
    CenterSpatialCropd,
    Invertd,
    OneOf,
    Orientationd,
    MapTransform,
    NormalizeIntensityd,
    RandSpatialCropSamplesd,
    CenterSpatialCropd,
    RandSpatialCropd,
    SpatialPadd,
    ScaleIntensityRanged,
    Spacingd,
    RepeatChanneld,
    ToTensord,
    ConvertToMultiChannelBasedOnBratsClassesd,
    RandFlipd,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    Resize
)
import torchvision


def get_bounding_box(ground_truth_map):
    '''
    This function creates varying bounding box coordinates based on the segmentation contours as prompt for the SAM model
    The padding is random int values between 5 and 20 pixels
    '''

    if len(np.unique(ground_truth_map)) > 1:

        # get bounding box from mask
        y_indices, x_indices = np.where(ground_truth_map > 0)
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)

        # add perturbation to bounding box coordinates
        H, W = ground_truth_map.shape
        x_min = max(0, x_min - np.random.randint(5, 20))
        x_max = min(W, x_max + np.random.randint(5, 20))
        y_min = max(0, y_min - np.random.randint(5, 20))
        y_max = min(H, y_max + np.random.randint(5, 20))

        bbox = [x_min, y_min, x_max, y_max]

        return bbox
    else:
        return [0, 0, 0, 0] # if there is no mask in the array, set bbox to 0


supported_tasks = ['detection', 'semantic_seg', 'instance_seg']
parser = argparse.ArgumentParser()
parser.add_argument('--task_name', default='semantic_seg', type=str)
parser.add_argument('--cfg', default='./finetune-anything/config/modelconfig.yaml', type=str)

import monai

if __name__ == '__main__':
    args = parser.parse_args()
    task_name = args.task_name
    if args.cfg is not None:
        config = OmegaConf.load(args.cfg)
    else:
        assert task_name in supported_tasks, "Please input the supported task name."
        config = OmegaConf.load("./config/{task_name}.yaml".format(task_name=args.task_name))

    train_cfg = config.train

    print("train cfg", train_cfg)
    # val_cfg = config.val
    # test_cfg = config.test
    train_f = open("train_dataloader_newtest.pkl",'rb')
    val_f = open("val_dataloader_newtest.pkl",'rb')


    train_loader = dill.load(train_f)
    val_loader = dill.load(val_f)

    print(args.cfg, train_cfg.losses)
    print(train_cfg.losses['DiceCE'])
    
    
    losses = get_losses(losses=train_cfg.losses)

    
    print("losses are: ", losses)

    # according the model name to get the adapted model
    
    model = get_model(model_name=train_cfg.model.sam_name, **train_cfg.model.params)

    opt_params = get_opt_pamams(model, lr_list=train_cfg.opt_params.lr_list, group_keys=train_cfg.opt_params.group_keys,
                                wd_list=train_cfg.opt_params.wd_list)
    
    # optimizer = get_optimizer(opt_name=train_cfg.opt_name, params=opt_params, lr=train_cfg.opt_params.lr_default,
    #                           momentum=train_cfg.opt_params.momentum, weight_decay=train_cfg.opt_params.wd_default)
    
    optimizer = get_optimizer(opt_name=train_cfg.opt_name, params=opt_params, lr=train_cfg.opt_params.lr_default,
                              weight_decay=train_cfg.opt_params.wd_default)


    scheduler = get_scheduler(optimizer=optimizer, lr_scheduler=train_cfg.scheduler_name)
    
    runner = get_runner(train_cfg.runner_name)(model, optimizer, losses, train_loader, val_loader, scheduler)
    # train_step

    runner.train(train_cfg)

    # if test_cfg.need_test:
    #     runner.test(test_cfg)
