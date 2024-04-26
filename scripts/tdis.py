import argparse, os, sys, glob
from statistics import mode
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
import PIL
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor


# load safety model
safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)


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
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept

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

def main():
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
        default="outputs/tdis-final-all-supp"
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
        default="configs/touch/tdis/eval_vision_conditional_on_touch_CMC_attn_layer5.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="/nfs/turbo/coe-ahowens/fredyang/stable-diffusion/logs/2023-11-13T14-06-19_recover_vision_conditional_on_touch_CMC_attn_layer5/checkpoints/epoch=000009.ckpt",
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
        "--strength",
        type=float,
        default=0.5,
        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image",
    )
    parser.add_argument(
        "--max_sample",
        type=float,
        default=30000,
        help="maximum number of sample",
    )

    opt = parser.parse_args()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        raise NotImplementedError("PLMS sampler not (yet) supported")
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    # 添加水印
    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()
    print("#### Data #####")
    # sample part of the data
    if opt.max_sample > 0:
        data.datasets['test']._length = opt.max_sample
    
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # 要改
    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else (batch_size + 4)
    # 移植到后面
    # if not opt.from_file:
    #     prompt = opt.prompt
    #     assert prompt is not None
    #     data = [batch_size * [prompt]]

    # else:
    #     print(f"reading prompts from {opt.from_file}")
    #     with open(opt.from_file, "r") as f:
    #         data = f.read().splitlines()
    #         data = list(chunk(data, batch_size))

    full_images_path = os.path.join(outpath, "full_images")
    os.makedirs(full_images_path, exist_ok=True)
    sample_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
    
    sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

    assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(opt.strength * opt.ddim_steps)
    print(f"target t_enc is {t_enc} steps")
    
    def load_test_img(path):
        image = Image.open(path).convert("RGB")

        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        
        image = image.resize((256, 256), resample=self.interpolation)
        

        # image = self.flip(image)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.*image - 1.

    with torch.no_grad():

        with precision_scope("cuda"):
            with model.ema_scope():
                for i in trange(len(data.datasets['test']), desc="Data"):
                    one_data = data.datasets['test'][i]
                    tic = time.time()
                    all_samples = list()
                    cond_key = model.cond_stage_key
                    #prompts_lists = [one_data['gelsightB'].to(device)]
                    prompts_lists = [load_test_img("/home/chfeng/BindDiffusion/tdis/leather/touch.jpg").to(device)]

                    #image_A = one_data['imageA'].to(device)
                    image_A = load_test_img("/home/chfeng/BindDiffusion/tdis/leather/input.png").to(device)
                    image_A = repeat(image_A, '1 ... -> b ...', b=batch_size)
                    image_A = model.get_first_stage_encoding(model.encode_first_stage(image_A))


                    # prompts_lists = [one_data[cond_key].to(device) for _ in range(opt.n_samples)]
                    # prompts_lists = [opt.n_samples * [one_data[cond_key].to(device)]]
                    # print(len(prompts_lists[0]))
                    # exit()
                    for n in trange(opt.n_iter, desc="Sampling"):
                        for prompts in tqdm(prompts_lists, desc="within_datapoint"):
                            uc = None
                            # 暂时设置为1.0
                            if opt.scale != 1.0:
                                print('unconditional not needed now!')
                                exit()
                                uc = model.get_learned_conditioning(batch_size * [""])
                            # if isinstance(prompts, tuple):
                            #     prompts = list(prompts)
                            prompts = torch.cat([prompts for _ in range(opt.n_samples)])
                            c = model.get_learned_conditioning(prompts)
                            # shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                            shape = (model.channels, model.image_size, model.image_size)

                            z_enc = sampler.stochastic_encode(image_A, torch.tensor([t_enc]*batch_size).to(device))
                            
                            samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,)
                            x_samples_ddim = model.decode_first_stage(samples)
                            x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                            x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                            # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)
                            # 跳过安全检查！
                            x_checked_image = x_samples_ddim

                            x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                            # save original images
                            if not opt.skip_save:
                                sample_path = os.path.join(full_images_path, f"{sample_count:04}")
                                os.makedirs(sample_path, exist_ok=True)

                                origin_images = [map_back(one_data["imageA"]), map_back(one_data["gelsightA"]), map_back(one_data["imageB"]), map_back(one_data["gelsightB"])]
                                origin_name = ['imageA', 'gelsightA', 'imageB', 'gelsightB']

                                for index, x_sample in enumerate(origin_images):
                                    x_sample = 255. * rearrange(x_sample.squeeze().cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    # img = put_watermark(img, wm_encoder)
                                    img.save(os.path.join(sample_path, "{}.png".format(origin_name[index])))


                                generated_image_count = 0
                                for x_sample in x_checked_image_torch:
                                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                    img = Image.fromarray(x_sample.astype(np.uint8))
                                    # img = put_watermark(img, wm_encoder)
                                    img.save(os.path.join(sample_path, f"{generated_image_count:05}.png"))
                                    generated_image_count += 1

                            if not opt.skip_grid:
                                all_samples.append(x_checked_image_torch)
                            

                    if not opt.skip_grid:
                        # additionally, save as grid

                        # 把原图和gelsight放进去
                        # origin_image = torch.cat((map_back(one_data[model.first_stage_key]), map_back(one_data[model.cond_stage_key])), 0)
                        origin_image = torch.cat([map_back(one_data["imageA"]), map_back(one_data['gelsightA']), map_back(one_data["imageB"]), map_back(one_data['gelsightB'])], 0)
                        all_samples = [torch.cat((origin_image, all_samples[k]), 0) for k in range(len(all_samples))]
                        grid = torch.stack(all_samples, 0)
                        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                        grid = make_grid(grid, nrow=n_rows)

                        # to image
                        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                        img = Image.fromarray(grid.astype(np.uint8))
                        img = put_watermark(img, wm_encoder)
                        img.save(os.path.join(outpath, f'grid-{sample_count:04}.png'))
                    
                    sample_count += 1

                    toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()