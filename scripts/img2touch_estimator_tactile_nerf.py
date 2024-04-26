import argparse
import os
import sys
import glob
from statistics import mode
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
from ipdb import set_trace as st

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
# safety_model_id = "CompVis/stable-diffusion-safety-checker"
# safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
# safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open(
            "assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


# def check_safety(x_image):
#     safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
#     x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
#     assert x_checked_image.shape[0] == len(has_nsfw_concept)
#     for i in range(len(has_nsfw_concept)):
#         if has_nsfw_concept[i]:
#             x_checked_image[i] = load_replacement(x_checked_image[i])
#     return x_checked_image, has_nsfw_concept

def get_input(batch):
    x = batch
    if len(x.shape) == 3:
        x = x[..., None]
    # x = rearrange(x, 'b h w c -> b c h w')
    x = x.to(memory_format=torch.contiguous_format).float()
    return x


def map_back(image):
    image = 127.5 * (image + 1.0) / 255.0
    # .astype(np.float32) / 255.0
    return image


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a painting of a virus monster playing guitar",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/img2touch-ycb/"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=3,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        # default=7.5,
        default=1.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="/home/fredyang/fredyang/stable-diffusion/logs/2023-10-13T11-56-07_img2touch_cmc_ae/configs/2023-10-13T11-56-07-project.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="logs/2023-10-13T11-56-07_img2touch_cmc_ae/checkpoints/last.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--max_sample",
        type=float,
        default=10000,
        help="maximum number of sample",
    )

    opt = parser.parse_args()
    return opt

class TouchEstimator:
    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.load_model()

    def load_model(self):
        config = OmegaConf.load(f"{self.opt.config}")
        model = load_model_from_config(config, f"{self.opt.ckpt}")
        model = model.to(self.device)
        if self.opt.plms:
            sampler = PLMSSampler(model)
        else:
            sampler = DDIMSampler(model)
        self.model = model
        self.sampler = sampler
    
    
    def load_rgbdb(self, rgb_path, depth_path, bg_path):
        # load rgb
        rgb = Image.open(rgb_path).convert("RGB")
        rgb = np.array(rgb).astype(np.uint8)
        crop = min(rgb.shape[0], rgb.shape[1])
        h, w, = rgb.shape[0], rgb.shape[1]
        rgb = rgb[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        rgb = Image.fromarray(rgb)
        # rgb = transforms.functional.rotate(rgb, angle=90)
        rgb = rgb.resize((256, 256), resample=PIL.Image.BICUBIC)
        rgb = np.array(rgb).astype(np.float32) / 255.0
        rgb = rgb[None].transpose(0, 3, 1, 2)
        rgb = torch.from_numpy(rgb)
        rgb = 2. * rgb - 1.
        
        # load depth
        depth_map = np.load(depth_path)
        depth_map = np.clip(depth_map, 0, 5)
        crop = min(depth_map.shape[0], depth_map.shape[1])
        h, w, = depth_map.shape[0], depth_map.shape[1]
        depth_map = depth_map[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2,:]
        # depth_map = np.rot90(depth_map)
        depth_map = cv2.resize(depth_map, (256, 256))
        depth_map = depth_map.reshape(1, 1, 256, 256).astype(np.float32)
        depth_map = torch.from_numpy(depth_map)
        
        # load bg
        bg = Image.open(bg_path).convert("RGB")
        bg = np.array(bg).astype(np.uint8)
        crop = min(bg.shape[0], bg.shape[1])
        h, w, = bg.shape[0], bg.shape[1]
        bg = bg[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]
        bg = Image.fromarray(bg)
        bg = transforms.functional.rotate(bg, angle=90)
        bg = bg.resize((256, 256), resample=PIL.Image.BICUBIC)
        bg = np.array(bg).astype(np.float32) / 255.0
        bg = bg[None].transpose(0, 3, 1, 2)
        bg = torch.from_numpy(bg)
        bg = 2. * bg - 1.
        
        rgbdb = torch.cat((rgb, depth_map, bg), 1)
        
        return rgbdb
    

    def estimate(self, rgb_dir, depth_dir, bg_path):
        rgb_paths = sorted(os.listdir(rgb_dir))
        depth_paths = sorted(os.listdir(depth_dir))
        depth_paths = [i for i in depth_paths if '.npy' in i]
        rgb_paths = [os.path.join(rgb_dir, rgb_path) for rgb_path in rgb_paths]
        depth_paths = [os.path.join(depth_dir, depth_path) for depth_path in depth_paths]
        
        os.makedirs(self.opt.outdir, exist_ok=True)
        outpath = self.opt.outdir
        full_images_path = os.path.join(outpath, "full_images")
        os.makedirs(full_images_path, exist_ok=True)

        start_code = None
        if self.opt.fixed_code:
            start_code = torch.randn(
                [self.opt.n_samples, self.opt.C, self.opt.H // self.opt.f, self.opt.W // self.opt.f], device=self.device)

        precision_scope = autocast if self.opt.precision == "autocast" else nullcontext
        with torch.no_grad():
            with precision_scope("cuda"):
                with self.model.ema_scope():
                    for rgb_path, depth_path in tqdm(zip(rgb_paths, depth_paths), total=len(rgb_paths)):
                        uc = None
                        rgbdb = self.load_rgbdb(rgb_path, depth_path, bg_path).to(self.device)
                        estimate_index = int(rgb_path.split('/')[-1].split('.')[0])
                        prompts = torch.cat([rgbdb for _ in range(self.opt.n_samples)])
                        c = self.model.get_learned_conditioning(prompts)
                        shape = (self.model.channels, self.model.image_size, self.model.image_size)
                        samples_ddim, _ = self.sampler.sample(S=self.opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=self.opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=self.opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=self.opt.ddim_eta,
                                                        x_T=start_code)
                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        x_checked_image = x_samples_ddim

                        x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)
                        
                        if not self.opt.skip_save:
                            sample_path = os.path.join(full_images_path, f"{estimate_index:04}")
                            os.makedirs(sample_path, exist_ok=True)
                            
                            origin_images = [map_back(rgbdb[:,:3])]
                            origin_name = ['rgb']

                            for index, x_sample in enumerate(origin_images):
                                x_sample = 255. * rearrange(x_sample.squeeze().cpu().numpy(), 'c h w -> h w c')
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                
                                img.save(os.path.join(sample_path, "{}.png".format(origin_name[index])))
                            
                            generated_image_count = 0
                            for x_sample in x_checked_image_torch:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                bg = rearrange(rgbdb[0,4:].cpu().numpy(), 'c h w -> h w c')
                                bg = 255.*(bg + 1.0) / 2.0
                                diff = x_sample - bg
                                # diff[np.abs(diff.mean(-1)) < 1] = 0
                                # scale = 127.5 / np.max(np.abs(diff))
                                scale = 0.8
                                x_sample = np.clip(127.5 + scale * diff, 0, 255)
                                # x_sample = (x_sample - x_sample.min()) / (x_sample.max() - x_sample.min()) * 255
                                print(x_sample.max(), x_sample.min(), scale)
                                # x_sample = np.clip(np.abs(x_sample-bg), 0, 255)
                                img = Image.fromarray(x_sample.astype(np.uint8))
                                img.save(os.path.join(sample_path, f"{generated_image_count:02}.png"))
                                generated_image_count += 1

def main():
    opt = parse_args()
    seed_everything(opt.seed)
    touch_estimator = TouchEstimator(opt)
    
    rgb_dir = "/home/ymdou/datac_ymdou/TaRF/nerfstudio/outputs/real_time_estimation_cache/rgb"
    depth_dir = "/home/ymdou/datac_ymdou/TaRF/nerfstudio/outputs/real_time_estimation_cache/depth"
    bg_path = "/home/ymdou/datac_ymdou/TaRF/nerfstudio/vision_touch_pairs/touch_bg/andrew_office_4_colmap_40_30/bg.jpg"
    
    touch_estimator.estimate(rgb_dir, depth_dir, bg_path)


if __name__ == "__main__":
    main()
