from __future__ import division
from __future__ import print_function

import argparse
import random
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import tools.utils as utils

################################################################################
# Infer CAM image, CAM dict and CRF dict from given checkpoints.
# All of the result files will be saved under experiment folder.
#
# To get CAM dict files...
# python infer.py --name [exp_name] --load_epo [epoch] --dict
#
# To get CRF dict files with certain alpha (let, a1 and a2)...
# python infer.py --name [exp_name] --load_epo [epoch] --crf --alphas a1 a2
#
# Of course you can do them at the same time.
# To get CAM image, simply add --vis option.
################################################################################

if __name__ == '__main__':

    categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                  'bus', 'car', 'cat', 'chair', 'cow', 
                  'diningtable', 'dog', 'horse', 'motorbike', 'person', 
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    parser = argparse.ArgumentParser()

    # Dataset
    parser.add_argument("--infer_list", default="train", type=str)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--batch_size", default=8, type=int)

    # Learning rate
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--wt_dec", default=5e-4, type=float) 

    # Experiments
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--seed", default=4242, type=int)
    parser.add_argument("--load_epo", required=True, type=int)

    # Output
    parser.add_argument("--vis", action='store_true')
    parser.add_argument("--dict", action='store_true')
    parser.add_argument("--crf", action='store_true')
    parser.add_argument("--alphas", default=[6,10,24], nargs='+', type=int)   

    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    args.max_step = 1

    print('Infer experiment ' + args.name + '!')
    exp_path, ckpt_path, train_path, val_path, infer_path, dict_path, crf_path, _ = utils.make_path(args)

    infer_dataset = utils.build_dataset(phase='val', path='voc12/'+args.infer_list+'.txt')
    infer_data_loader = DataLoader(infer_dataset, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print('Infer dataset is loaded from ' + args.infer_list)
    
    model = getattr(importlib.import_module('models.'+args.model), 'model_WSSS')(args)
    model.load_model(args.load_epo, ckpt_path)
    model.set_phase('eval')

    model.infer_init()
    print('-'*111)
    print(('-'*46)+' Start infer loop '+('-'*47))
    print('-'*111)
    
    for iter, pack in enumerate(tqdm(infer_data_loader)):

        model.unpack(pack)
        model.infer_msf(0, infer_path, dict_path, crf_path, vis=args.vis, dict=args.dict, crf=args.crf)