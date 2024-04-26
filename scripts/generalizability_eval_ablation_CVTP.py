import json
import os
import os.path as osp
import numpy as np

from tqdm import tqdm
from scipy.spatial import KDTree
import shutil
import concurrent.futures
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms as T
from PIL import Image
from pytorch_msssim import ssim
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import math
import seaborn as sns
import pandas as pd


scene_list = [
    'eecs_coffeeroom_colmap',
    'eecs_office_2_colmap',
    'eecs_3312_colmap',
    'eecs_3313_colmap',
    'eecs_3414_colmap',
    'eecs_conference_room_colmap',
    'andrew_office_large_2_colmap',
    'andrew_office_4_colmap',
    'andrew_office_5_colmap',
    'eecs_room_colmap',
    'eecs_corridor_colmap',
]

data_dir = '/home/ymdou/stable-diffusion/tactile_nerf/vision_touch_pairs_tactile_nerf_final'


def load_camera_as_pcd(scene_list):
    camera_path_dir = osp.join(data_dir, 'camera_path')
    scene2pcd = {}
    for scene in tqdm(scene_list, leave=False):
        camera_path_file = osp.join(camera_path_dir, scene+'.json')
        camera_path = json.load(open(camera_path_file, 'r'))['camera_path']
        cur_pcd = []
        for cam in camera_path:
            cur_pcd.append(cam['camera_to_world'])
        cur_pcd = np.array(cur_pcd)
        cur_pcd = cur_pcd[:, :3, 3]
        scene2pcd[scene] = cur_pcd

    return scene2pcd


def load_split_as_idx(split_path):
    scene2idx = {
        scene: {'train': [], 'val': [], 'test': []} for scene in scene_list
    }
    split = json.load(open(split_path, 'r'))
    for k in split:
        for inst in split[k]:
            cur_scene = inst[0].replace('_40_30', '')
            scene2idx[cur_scene][k].append(int(inst[1]))
        for scene in scene_list:
            scene2idx[scene][k] = np.array(scene2idx[scene][k])

    return scene2idx


def build_kdtree(scene2pcd, scene2idx):
    scene2kdtree = {}
    for scene in tqdm(scene_list, leave=False):
        cur_pcd = scene2pcd[scene]
        cur_train_idx = scene2idx[scene]['train']
        cur_test_idx = scene2idx[scene]['test']
        if cur_test_idx.shape[0] == 0:
            continue
        cur_train_pcd = cur_pcd[cur_train_idx]
        tree = KDTree(cur_train_pcd)
        scene2kdtree[scene] = tree
    return scene2kdtree


class ResnetEncoder(nn.Module):
    def __init__(self,
                 model_name='resnet18',):
        super().__init__()
        if model_name == 'resnet18':
            self.resnet = models.resnet18(True)
            self.resnet.fc = nn.Linear(512, 16)
        elif model_name == 'resnet50':
            self.resnet = models.resnet50(True)

    def forward(self, batch: torch.tensor):
        '''
        takes in an image and returns the resnet18 features
        '''
        features = self.resnet(batch)
        feat_norm = torch.norm(features, dim=1)
        return features/feat_norm.view(features.shape[0], 1)


def calc_psnr(img1, img2):
    if len(img1.shape) == 3:
        mse = torch.mean((img1 - img2) ** 2)
        return 20. * torch.log10(1. / torch.sqrt(mse))
    elif len(img1.shape) == 4:
        mse = torch.mean((img1 - img2) ** 2, dim=[1, 2, 3])
        psnr = 20. * torch.log10(1. / torch.sqrt(mse))
        return torch.mean(psnr)


def calc_ssim(img1, img2):
    # img1: (N,3,H,W) a batch of RGB images
    # img2: (N,3,H,W)
    ssim_val = ssim(img1, img2, data_range=1.)
    return ssim_val


if __name__ == '__main__':
    # CLIP
    preprocess = {
        'input': T.Compose([
            T.Resize(128),
            T.Grayscale(3),
            T.ToTensor(),
        ]),
        'tac': T.Compose([
            T.Resize(128),
            T.ToTensor(),
        ])
    }
    rgb_enc = ResnetEncoder()
    tac_enc = ResnetEncoder()
    rgb_enc.load_state_dict(torch.load(
        '/home/ymdou/stable-diffusion/logs/ckpts/ResnetEncoder/rgb_enc.pth', map_location='cpu'))
    # '/home/ymdou/stable-diffusion/logs/ckpts/ResnetEncoder/rgb_enc_with_test.pth', map_location='cpu'))
    tac_enc.load_state_dict(torch.load(
        '/home/ymdou/stable-diffusion/logs/ckpts/ResnetEncoder/tac_enc.pth', map_location='cpu'))
    # '/home/ymdou/stable-diffusion/logs/ckpts/ResnetEncoder/tac_enc_with_test.pth', map_location='cpu'))
    rgb_enc.cuda()
    tac_enc.cuda()

    rgb_enc.eval()
    tac_enc.eval()

    scene2idx = load_split_as_idx(osp.join(data_dir, 'split_interval.json'))
    scene2pcd = load_camera_as_pcd(scene_list)
    scene2kdtree = build_kdtree(scene2pcd, scene2idx)

    orig_gt = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/gt'
    orig_pred_diffusion = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/pred'
    orig_input = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/full_images'
    dst = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/bins'

    split = json.load(open(osp.join(data_dir, 'split_interval.json'), 'r'))
    dist_score_pairs = []
    for instance_idx, instance in tqdm(enumerate(split['test']),total=len(split['test']), leave=False):
        gt_fname = osp.join(orig_gt, f'{str(instance_idx).zfill(4)}.png')
        pred_diffusion_fname = osp.join(orig_pred_diffusion, f'{str(instance_idx).zfill(4)}.png')

        input_fname = osp.join(orig_input, f'{str(instance_idx).zfill(4)}', 'input.png')

        input = preprocess['input'](Image.open(input_fname)).cuda()
        pred_diffusion = preprocess['tac'](Image.open(pred_diffusion_fname)).cuda()

        with torch.no_grad():
            input_feat = rgb_enc(input.unsqueeze(0)).squeeze(0)
            pred_diffusion_feat = tac_enc(pred_diffusion.unsqueeze(0)).squeeze(0)

        dist_score_pairs.append(
                torch.dot(input_feat, pred_diffusion_feat).cpu().detach().numpy(),
        )
    dist_score_pairs = np.array(dist_score_pairs)
    print('CVTP: {:.2f}'.format(dist_score_pairs.mean()))
