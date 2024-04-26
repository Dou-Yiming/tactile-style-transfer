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
    orig_pred_gan = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/pred_gan_new'
    orig_pred_l1 = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/pred_l1_new'
    orig_input = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/full_images'
    dst = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/bins'

    split = json.load(open(osp.join(data_dir, 'split_interval.json'), 'r'))
    dist_score_pairs = []
    for instance_idx, instance in tqdm(enumerate(split['test']),total=len(split['test']), leave=False):
        scene = instance[0].replace('_40_30', '')
        idx = int(instance[1])
        cur_pcd = scene2pcd[scene]
        cur_test_pcd = cur_pcd[idx]
        dist, _ = scene2kdtree[scene].query(cur_test_pcd)

        gt_fname = osp.join(orig_gt, f'{str(instance_idx).zfill(4)}.png')
        pred_diffusion_fname = osp.join(orig_pred_diffusion, f'{str(instance_idx).zfill(4)}.png')
        pred_gan_fname = osp.join(orig_pred_gan, f'{str(instance_idx).zfill(4)}.png')
        pred_l1_fname = osp.join(orig_pred_l1, f'{str(instance_idx).zfill(4)}.png')

        input_fname = osp.join(
            orig_input, f'{str(instance_idx).zfill(4)}', 'input.png')

        gt = preprocess['tac'](Image.open(gt_fname)).cuda()
        pred_diffusion = preprocess['tac'](Image.open(pred_diffusion_fname)).cuda()
        pred_gan = preprocess['tac'](Image.open(pred_gan_fname)).cuda()
        pred_l1 = preprocess['tac'](Image.open(pred_l1_fname)).cuda()

        dist_score_pairs.append(
            [
                dist,
                [
                    calc_psnr(pred_diffusion, gt).cpu().detach().numpy(),
                    calc_psnr(pred_gan, gt).cpu().detach().numpy(),
                    calc_psnr(pred_l1, gt).cpu().detach().numpy(),
                ]
            ]
        )
    dist_score_pairs = np.array(dist_score_pairs, dtype=object)
    np.save('dist_score.npy', dist_score_pairs)
    dist_score_pairs = np.load('dist_score.npy', allow_pickle=True)
    bin_number = 3
    bin_step = (dist_score_pairs[:, 0].max() -
                dist_score_pairs[:, 0].min()) / bin_number + 1e-8
    score_bin = [
        [] for _ in range(bin_number)
    ]
    for dist_score_pair in dist_score_pairs:
        bin_idx = math.floor(
            (dist_score_pair[0]-dist_score_pairs[:, 0].min())/bin_step)
        score_bin[bin_idx].append(
            [float(dist_score_pair[1][0]), float(
                dist_score_pair[1][1]), float(dist_score_pair[1][2])]
        )
    print([len(bin) for bin in score_bin])
    score_bin_mean = [list(np.array(bin).mean(0)) for bin in score_bin]
    score_bin_std = np.array([list(np.array(bin).std(0)) for bin in score_bin])
    score_bin_se = score_bin_std
    for i, bin in enumerate(score_bin):
        score_bin_se[:,i]/=np.sqrt(len(bin))
    

    # Create a DataFrame
    data = {
        'Diffusion': score_bin_mean[0],
        'GAN': score_bin_mean[1],
        'L1': score_bin_mean[2],
    }
    from pprint import pprint
    pprint(data)
    data_se = {
        'Diffusion_SE': score_bin_se[0],
        'GAN_SE': score_bin_se[1],
        'L1_SE': score_bin_se[2],
    }

    categories = ['[{:.2f}, {:.2f})'.format(0, bin_step*100),
                  '[{:.2f}, {:.2f})'.format(bin_step*100, bin_step*200),
                  '[{:.2f}, {:.2f})'.format(bin_step*200, bin_step*300)]
    df = pd.DataFrame(data, index=categories)
    df_se = pd.DataFrame(data_se, index=categories)

    # Melt the DataFrame
    df_melted = df.reset_index().melt(id_vars='index')
    # Create the bar plot
    plt.figure(figsize=(10, 8))
    ax = sns.barplot(x='index', y='value', hue='variable',
                     data=df_melted, palette='Spectral', saturation=0.8, dodge=True)

    n_groups = len(df.index)
    n_categories = len(df.columns)
    bar_width = 0.2
    for i, col in enumerate(df.columns):
        errs = df_se[col + "_SE"]
        x_coords = [p.get_x() + p.get_width() / 2. for p in ax.patches[i*3:i*3+3]]
        ax.errorbar(x_coords, df[col], yerr=errs,
                    fmt='none', capsize=5, color='black')

    # Rename the axes
    plt.xlabel('Distance (cm)', fontsize=14)
    plt.ylabel('PSNR', fontsize=14)

    # plt.ylim(20)

    # Set the legend title
    plt.legend(title='', loc="upper left",
               ncol=1, frameon=False, fontsize=14)

    plt.xticks(rotation=0, fontsize=14)
    plt.yticks(rotation=0, fontsize=14)
        
    # Show the plot
    plt.margins(0.03, tight=True)
    plt.savefig('PSNR_hist.png', dpi=300, bbox_inches='tight')
