import os
import shutil
from tqdm import tqdm


input_path = '/home/fredyang/fredyang/pytorch-CycleGAN-and-pix2pix/results/pix2pix_img2touch'
inter_path = 'test_latest/images'
output_num = 1
raw_img = os.path.join(input_path, 'test_latest/images')

os.makedirs(os.path.join(input_path, 'eval'), exist_ok = True)
eval_dir = os.path.join(input_path, 'eval')

os.makedirs(os.path.join(input_path, 'eval', 'source'), exist_ok = True)
eval_dir_source = os.path.join(input_path, 'eval', 'source')
os.makedirs(os.path.join(input_path, 'eval', 'target'), exist_ok = True)
eval_dir_traget = os.path.join(input_path, 'eval', 'target')

for i, pic in enumerate(tqdm(os.listdir(raw_img))):
# for folder in os.listdir(raw_img):
    # print(folder)
    # print(os.path.join(str(eval_dir_source), str(folder) + '.png'))
    # move source
    number = pic[:-11]
    back = pic[-11:]
    if back == '_fake_B.png':
        real_img = number + '_real_B.png'
        shutil.copy(os.path.join(str(raw_img), pic), os.path.join(str(eval_dir_traget), pic))
        shutil.copy(os.path.join(str(raw_img), real_img), os.path.join(str(eval_dir_source), pic))
        # print(number, back)
        # exit()