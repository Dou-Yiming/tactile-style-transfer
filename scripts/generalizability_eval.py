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
    scene2idx = load_split_as_idx(osp.join(data_dir, 'split_interval.json'))
    scene2pcd = load_camera_as_pcd(scene_list)
    scene2kdtree = build_kdtree(scene2pcd, scene2idx)

    orig_gt = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/gt'
    orig_pred = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/pred'
    # orig_pred = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/pred_gan_new'
    orig_input = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/full_images'
    dst = '/home/ymdou/stable-diffusion/outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/bins'

    split = json.load(open(osp.join(data_dir, 'split_interval.json'), 'r'))
    dist_all = []
    for instance_idx, instance in enumerate(split['test']):
        scene = instance[0].replace('_40_30', '')
        idx = int(instance[1])
        cur_pcd = scene2pcd[scene]
        cur_test_pcd = cur_pcd[idx]
        dist, _ = scene2kdtree[scene].query(cur_test_pcd)
        dist_all.append(dist)
    dist_all = np.array(dist_all)
    dist_all.sort()
    # import ipdb; ipdb.set_trace()
    bin_numbers = 5
    bin_values = [0] + [dist_all[len(dist_all) * i // bin_numbers - 1]
                        for i in range(1, bin_numbers + 1)]
    # bin_values = [0, 0.2, 0.4, 0.6, 0.8, 1]
    print(bin_values)

    def process_instance(instance_idx, instance, scene2pcd, scene2kdtree, bin_values, orig_gt, orig_pred, dst):
        scene = instance[0].replace('_40_30', '')
        idx = int(instance[1])
        cur_pcd = scene2pcd[scene]
        cur_test_pcd = cur_pcd[idx]
        dist, _ = scene2kdtree[scene].query(cur_test_pcd)
        for i in range(1, len(bin_values)):
            dst_gt = osp.join(dst, f'{i}', 'gt')
            dst_pred = osp.join(dst, f'{i}', 'pred')
            dst_input = osp.join(dst, f'{i}', 'input')
            os.makedirs(dst_gt, exist_ok=True)
            os.makedirs(dst_pred, exist_ok=True)
            os.makedirs(dst_input, exist_ok=True)
            if dist < bin_values[i]:
                shutil.copy(osp.join(orig_gt, f'{str(instance_idx).zfill(4)}.png'), osp.join(
                    dst_gt, f'{str(instance_idx).zfill(4)}.png'))
                shutil.copy(osp.join(orig_pred, f'{str(instance_idx).zfill(4)}.png'), osp.join(
                    dst_pred, f'{str(instance_idx).zfill(4)}.png'))
                shutil.copy(osp.join(orig_input, f'{str(instance_idx).zfill(4)}', 'input.png'), osp.join(
                    dst_input, f'{str(instance_idx).zfill(4)}.png'))
                break

    def process_instances(scene2pcd, scene2kdtree, bin_values, orig_gt, orig_pred, dst):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for instance_idx, instance in enumerate(split['test']):
                futures.append(executor.submit(process_instance, instance_idx, instance,
                               scene2pcd, scene2kdtree, bin_values, orig_gt, orig_pred, dst))
            concurrent.futures.wait(futures)

    process_instances(scene2pcd, scene2kdtree,
                      bin_values, orig_gt, orig_pred, dst)

    # FID
    # for i in range(1, len(bin_values)):
    #     dst_gt = osp.join(dst, f'{i}', 'gt')
    #     dst_pred = osp.join(dst, f'{i}', 'pred')
    #     print(f'bin: {i}, distance: [{bin_values[i-1]}, {bin_values[i]}), files: {len(os.listdir(dst_gt))}')
    #     os.system(f'python -m pytorch_fid {dst_gt} {dst_pred} --device cuda:0')

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

    # compute CLIP similarity for <GT, input> and <pred, input> in each bin using the rgb and tac encoders
    bin2clip_score_pred = {}
    bin2clip_score_gt = {}
    bin2psnr = {}
    bin2ssim = {}
    import matplotlib.pyplot as plt

    fig, axs = plt.subplots(10, 10)
    # enlarge the figure size
    fig.set_size_inches(10,10)
    for i in range(1, len(bin_values)):
        bin2clip_score_pred[i] = []
        bin2clip_score_gt[i] = []
        bin2psnr[i] = []
        bin2ssim[i] = []
        input_dir = osp.join(dst, f'{i}', 'input')
        gt_dir = osp.join(dst, f'{i}', 'gt')
        pred_dir = osp.join(dst, f'{i}', 'pred')

        input_fnames = os.listdir(input_dir)
        gt_fnames = os.listdir(gt_dir)
        pred_fnames = os.listdir(pred_dir)

        input_fnames.sort()
        gt_fnames.sort()
        pred_fnames.sort()
        
        f_idx =0
        for input_fname, pred_fname, gt_fname in tqdm(zip(input_fnames[:10], pred_fnames[:10], gt_fnames[:10]), leave=False, total=len(input_fnames)):
            input_fname = osp.join(input_dir, input_fname)
            pred_fname = osp.join(pred_dir, pred_fname)
            gt_fname = osp.join(gt_dir, gt_fname)
            
            # add image to the subplot
            axs[(i-1)*2, f_idx].imshow(Image.open(gt_fname))
            axs[(i-1)*2+1, f_idx].imshow(Image.open(pred_fname))
            axs[(i-1)*2, f_idx].axis('off')
            axs[(i-1)*2+1, f_idx].axis('off')
            f_idx += 1

            input = preprocess['input'](Image.open(input_fname)).cuda()
            pred = preprocess['tac'](Image.open(pred_fname)).cuda()
            gt = preprocess['tac'](Image.open(gt_fname)).cuda()
            # import ipdb; ipdb.set_trace()
            # pass
            with torch.no_grad():
                input_feat = rgb_enc(input.unsqueeze(0)).squeeze(0)
                pred_feat = tac_enc(pred.unsqueeze(0)).squeeze(0)
                gt_feat = tac_enc(gt.unsqueeze(0)).squeeze(0)

            # import ipdb; ipdb.set_trace()
            
            bin2clip_score_pred[i].append(
                torch.dot(input_feat, pred_feat).cpu().detach().numpy())
            bin2clip_score_gt[i].append(
                torch.dot(input_feat, gt_feat).cpu().detach().numpy())
            bin2psnr[i].append(calc_psnr(pred, gt).cpu().detach().numpy())
            bin2ssim[i].append(calc_ssim(pred.unsqueeze(0), gt.unsqueeze(0)).squeeze().cpu().detach().numpy())
        bin2clip_score_pred[i] = np.array(bin2clip_score_pred[i])
        bin2clip_score_gt[i] = np.array(bin2clip_score_gt[i])
        bin2psnr[i] = np.array(bin2psnr[i])
        bin2ssim[i] = np.array(bin2ssim[i])
        print('bin: {}, distance: [{:.2f}, {:.2f}), files: {}, PSNR: {:.2f}, SSIM: {:.2f}, CLIP similarity (rgb-pred): {:.2f}, CLIP similarity (rgb-gt): {:.2f}'.format(
            i, bin_values[i-1], bin_values[i], len(os.listdir(gt_dir)),
            bin2psnr[i].mean(), bin2ssim[i].mean(),
            bin2clip_score_pred[i].mean(), bin2clip_score_gt[i].mean()))
    plt.tight_layout()
    # remove the x and y ticks
    plt.xticks([])
    plt.yticks([])
    # remove the axis
    plt.axis('off')
    plt.show()
    plt.savefig('bins.png')
    print('PSNR: {:.2f}, SSIM: {:.2f}'.format(
        np.concatenate(list(bin2psnr.values())).mean(), np.concatenate(list(bin2ssim.values())).mean()))
    print('CLIP similarity (rgb-pred): {:.2f}, CLIP similarity (rgb-gt): {:.2f}'.format(
        np.concatenate(list(bin2clip_score_pred.values())).mean(), np.concatenate(list(bin2clip_score_gt.values())).mean()))
    
    
    
    # os.system(f'rm -rf {dst}/*')
