import os
import os.path as osp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tools import utils, pyutils
from tools.imutils import save_img, denorm, _crf_with_alpha

# import resnet38d
from networks import resnet38d

def set_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad

class model_WSSS():

    def __init__(self, args):
        
        # Common things
        self.categories = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                           'bus', 'car', 'cat', 'chair', 'cow',
                           'diningtable', 'dog', 'horse', 'motorbike', 'person',
                           'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        self.args = args
        self.bs = args.batch_size
        self.phase = 'train'
        self.dev = 'cuda'

        # Instances
        self.net_names = ['net_cl', 'net_er']
        self.base_names = ['all', 'select']
        self.loss_names = ['loss_' + bn for bn in self.base_names]
        self.acc_names = ['acc_' + bn for bn in self.base_names]
        self.nets = []
        self.opts = []

        # Loss-related
        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.accs = [0] * len(self.acc_names)
        self.count = 0
        self.bce = nn.BCEWithLogitsLoss()       
        
        # Define networks
        self.net_er = resnet38d.Net_er()
        self.net_cl = resnet38d.Net_cl()
        
    # Save networks
    def save_model(self, epo, ckpt_path, best=None):
        epo_str = str(epo).zfill(3)
        if best:
            epo_str = 'best'
        torch.save(self.net_er.module.state_dict(), ckpt_path + '/' + epo_str + 'net_er.pth')
        torch.save(self.net_cl.module.state_dict(), ckpt_path + '/' + epo_str + 'net_cl.pth')

    def load_pretrained(self, er_path, cl_path):
        if er_path == 'imagenet':
            self.net_er.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)
        elif er_path == 'cam':
            self.net_er.load_state_dict(torch.load('./pretrained/od_cam.pth'), strict=True)
        if cl_path == 'imagenet':  
            self.net_cl.load_state_dict(resnet38d.convert_mxnet_to_torch('./pretrained/resnet_38d.params'), strict=False)  
        elif cl_path == 'cam':
            self.net_cl.load_state_dict(torch.load('./pretrained/od_cam.pth'), strict=True)

    # Load networks
    def load_model(self, epo, ckpt_path):
        epo_str = str(epo).zfill(3)
        self.net_er.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_er.pth'), strict=True)
        self.net_cl.load_state_dict(torch.load(ckpt_path + '/' + epo_str + 'net_cl.pth'), strict=True)
        self.net_er = torch.nn.DataParallel(self.net_er.to(self.dev))
        self.net_cl = torch.nn.DataParallel(self.net_cl.to(self.dev))

    # Set networks' phase (train/eval)
    def set_phase(self, phase):
        if phase == 'train':
            self.phase = 'train'
            for name in self.net_names:
                getattr(self, name).train()
        else:
            self.phase = 'eval'
            for name in self.net_names:
                getattr(self, name).eval()
        self.net_cl.eval()

    # Set optimizers and upload networks on multi-gpu
    def train_setup(self):

        args = self.args
        param_er = self.net_er.get_parameter_groups()

        self.opt_er = utils.PolyOptimizer([
            {'params': param_er[0], 'lr': 1 * args.lr, 'weight_decay': args.wt_dec},
            {'params': param_er[1], 'lr': 2 * args.lr, 'weight_decay': 0}, # non-scratch bias
            {'params': param_er[2], 'lr': 10 * args.lr, 'weight_decay': args.wt_dec}, # scratch weight
            {'params': param_er[3], 'lr': 20 * args.lr, 'weight_decay': 0} # scratch bias
        ],
            lr=args.lr, weight_decay=args.wt_dec, max_step=args.max_step)
        self.net_er = torch.nn.DataParallel(self.net_er.to(self.dev))
        self.net_cl = torch.nn.DataParallel(self.net_cl.to(self.dev))

        self.nets.append(self.net_er)
        self.nets.append(self.net_cl)

    # Unpack data pack from data_loader
    def unpack(self, pack):
        
        self.name = pack[0][0]

        if self.phase == 'train':
            self.img = pack[1].to(self.dev)
            self.label = pack[2].to(self.dev)

        if self.phase == 'eval':
            self.img = pack[1]
            # To handle MSF dataset
            for i in range(8):
                self.img[i] = self.img[i].to(self.dev)
            self.label = pack[2].to(self.dev)
    
        self.split_label()

    # Randomly select mask-class and remain-class and define labels for them
    def split_label(self):

        bs = self.label.shape[0]
        self.label_mask = torch.zeros(bs, 20).cuda()
        self.label_remain = self.label.clone()
        for i in range(bs):
            label_idx = torch.nonzero(self.label[i], as_tuple=False)
            rand_idx = torch.randint(0, len(label_idx), (1,))
            target = label_idx[rand_idx][0]
            self.label_remain[i, target] = 0
            self.label_mask[i, target] = 1

        self.label_all = self.label

    # Do forward/backward propagation and call optimizer to update the networks
    def update(self, epo):

        self.weight = self.args.cc_init+self.args.cc_slope*epo

        self.opt_er.zero_grad()
        self.cam_er, self.out_er = self.net_er(self.img)

        self.mask = self.cam_er[self.label_mask == 1, :, :].unsqueeze(1)
        self.mask = F.interpolate(self.mask, self.img.size()[2:], mode='bilinear', align_corners=False)
        self.mask = F.relu(self.mask)
        self.mask = self.mask / (self.mask.max() + 1e-5)
        self.img_masked = self.img * (1 - self.mask)
        self.cam_cl, self.out_cl = self.net_cl(self.img_masked)

        self.loss_all = self.bce(self.out_er, self.label_all)
        self.loss_select = self.weight*self.bce(self.out_cl, self.label_remain)

        loss = self.loss_all + self.loss_select
        loss.backward()
        self.opt_er.step()

        self.count_rw(self.out_er, self.label_all, 0)
        self.count_rw(self.out_cl, self.label_remain, 1)

        for i in range(len(self.loss_names)):
            self.running_loss[i] += getattr(self, self.loss_names[i]).item()
        self.count += 1

    # Count the number of right/wrong predictions for each accuracy
    def count_rw(self, out, label, idx):
        for b in range(self.bs):
            gt = label[b].cpu().detach().numpy()
            gt_cls = np.nonzero(gt)[0]
            num = len(np.nonzero(gt)[0])
            pred = out[b].cpu().detach().numpy()
            pred_cls = pred.argsort()[-num:][::-1]

            for c in gt_cls:
                if c in pred_cls:
                    self.right_count[idx] += 1
                else:
                    self.wrong_count[idx] += 1

    def infer_init(self):
        n_gpus = torch.cuda.device_count()
        self.net_er_replicas = torch.nn.parallel.replicate(self.net_er.module,list(range(n_gpus)))

    # (Multi-Thread) Infer MSF-CAM and save image/cam_dict/crf_dict 
    def infer_msf(self, epo, val_path, dict_path, crf_path, vis=False, dict=False, crf=False):

        if self.phase!='eval':
            self.set_phase('eval')

        epo_str = str(epo).zfill(3)
        gt = self.label_all[0].cpu().detach().numpy()
        self.gt_cls = np.nonzero(gt)[0]

        _, _, H, W = self.img[2].shape
        n_gpus = torch.cuda.device_count()

        def _work(i, img):
            with torch.no_grad():
                with torch.cuda.device(i % n_gpus):
                    cam,_ = self.net_er_replicas[i % n_gpus](img.cuda())
                    cam = F.upsample(cam, (H,W), mode='bilinear', align_corners=False)[0]
                    cam = F.relu(cam)
                    cam = cam.cpu().numpy() * self.label.clone().cpu().view(20, 1, 1).numpy()
                    
                    if i % 2 == 1:
                        cam = np.flip(cam, axis=-1)

                    return cam

        thread_pool = pyutils.BatchThreader(_work, list(enumerate(self.img)), batch_size=8, prefetch_size=0, processes=8)

        cam_list = thread_pool.pop_results()
        cam = np.sum(cam_list, axis=0)
        cam_max = np.max(cam, (1, 2), keepdims=True)
        norm_cam = cam / (cam_max + 1e-5)

        self.cam_dict = {}
        for i in range(20):
            if self.label[0, i] > 1e-5:
                self.cam_dict[i] = norm_cam[i]

        if vis:
            img_np = denorm(self.img[2][0]).cpu().detach().data.permute(1, 2, 0).numpy()
            for c in self.gt_cls:
                save_img(osp.join(val_path, epo_str + '_' + self.name + '_cam_' + self.categories[c] + '.png'), img_np, norm_cam[c])

        if dict:
            np.save(osp.join(dict_path, self.name + '.npy'), self.cam_dict)

        if crf:
            for a in self.args.alphas:
                crf_dict = _crf_with_alpha(self.cam_dict, self.name, alpha=a)
                np.save(osp.join(crf_path, str(a).zfill(2), self.name + '.npy'), crf_dict)

    # Print loss/accuracy (and re-initialize them)
    def print_log(self, epo, iter, logger):

        loss_str = ''
        acc_str = ''
        lr_str = ''

        for i in range(len(self.loss_names)):
            loss_str += self.base_names[i] + ' : ' + str(round(self.running_loss[i] / self.count, 5)) + ', '

        for i in range(len(self.acc_names)):
            if self.right_count[i]!=0:
                acc = 100 * self.right_count[i] / (self.right_count[i] + self.wrong_count[i])
                acc_str += self.acc_names[i] + ' : ' + str(round(acc, 2)) + ', '
                self.accs[i] = acc
        lr_str = 'Learning rate er:cl = 1:' + str(round(self.weight, 5))

        logger.info(loss_str[:-2])
        logger.info(acc_str[:-2])
        logger.info(lr_str)

        self.running_loss = [0] * len(self.loss_names)
        self.right_count = [0] * len(self.acc_names)
        self.wrong_count = [0] * len(self.acc_names)
        self.count = 0
