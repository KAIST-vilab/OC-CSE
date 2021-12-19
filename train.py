from __future__ import division
from __future__ import print_function

import os
import os.path as osp

import argparse
import logging

from tqdm import tqdm
import importlib

import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import tools.utils as utils

from evaluation import validation

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--train_list", default="train_aug", type=str)
    parser.add_argument("--val_list", default="train", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--resize", default=[256,448], nargs='+', type=int)
    parser.add_argument("--crop", default=384, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float)  
    parser.add_argument("--max_epochs", default=15, type=int)

    # Experiments
    parser.add_argument("--model", required=True, type=str) # model_cse
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--seed", default=4242, type=int)
    parser.add_argument("--er_init", default='imagenet', type=str)
    parser.add_argument("--cl_init", default='cam', type=str)
    parser.add_argument("--cc_init", default=0.3, type=float)
    parser.add_argument("--cc_slope", default=0.05, type=float)

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--print_freq", default=100, type=int)
    parser.add_argument("--vis_freq", default=100, type=int)
    parser.add_argument("--alphas", default=[6,10,24], nargs='+', type=int)    

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    print('Start experiment ' + args.name + '!')
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, log_path = utils.make_path(args)

    if osp.isfile(log_path):
        os.remove(log_path)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(log_path)
    logger.addHandler(file_handler)

    logger.info('-'*52 + ' SETUP ' + '-'*52)
    for arg in vars(args):
        logger.info(arg + ' : ' + str(getattr(args, arg)))
    logger.info('-'*111)

    train_dataset = utils.build_dataset(phase='train', path='voc12/'+args.train_list+'.txt', resize=args.resize, crop=args.crop)
    val_dataset = utils.build_dataset(phase='val', path='voc12/'+args.val_list+'.txt')
    train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_data_loader = DataLoader(val_dataset, shuffle=False, num_workers=0, pin_memory=True)

    logger.info('Train dataset is loaded from ' + 'voc12/'+args.train_list+'.txt')
    logger.info('Validation dataset is loaded from ' + 'voc12/'+args.val_list+'.txt')

    train_num_img = len(train_dataset)
    train_num_batch = len(train_data_loader)
    max_step = train_num_img // args.batch_size * args.max_epochs
    args.max_step = max_step
    max_miou = 0

    model = getattr(importlib.import_module('models.'+args.model), 'model_WSSS')(args)
    model.load_pretrained(args.er_init, args.cl_init)
    model.train_setup()

    logger.info('-'*111)
    logger.info(('-'*43)+' start OC-CSE train loop '+('-'*44))
    max_epo = 0
    max_miou = 0
    max_thres = 0
    max_list = []

    for epo in range(args.max_epochs):   
        
        # Train
        logger.info('-'*111)
        logger.info('Epoch ' + str(epo).zfill(3) + ' train')
        model.set_phase('train')
        for iter, pack in enumerate(tqdm(train_data_loader)):
            model.unpack(pack)
            model.update(epo)            
            if iter%args.print_freq==0 and iter!=0:
                model.print_log(epo, iter/train_num_batch, logger)
                logger.info('-')           
        model.save_model(epo, ckpt_path)

        # Validation
        logger.info('_'*111)
        logger.info('Epoch ' + str(epo).zfill(3) + ' validation')
        model.set_phase('eval')
        model.infer_init()
        for iter, pack in enumerate(tqdm(val_data_loader)): 
            model.unpack(pack)
            model.infer_msf(epo, val_path, dict_path, crf_path, vis=iter<50, dict=True, crf=False)
        
        miou, thres = validation('train', args.name, 'cam', 'dict', logger=logger)
        logger.info('Epoch ' + str(epo) + ' mIoU=' + str(miou)[:6] + '% at threshold ' + str(thres))
        max_list.append(miou)
        if miou>max_miou:
            max_miou = miou
            max_thres = thres
            max_epo = epo
        logger.info('Epoch ' + str(max_epo) + ' is best : mIoU=' + str(max_miou)[:6] + '% at threshold ' + str(max_thres))
        logger.info([round(vals,1) for vals in max_list])
