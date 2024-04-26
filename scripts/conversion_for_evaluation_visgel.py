import os
import shutil
from tqdm import tqdm


input_path = '/home/fredyang/fredyang/iccv_results/visgel_touch2img_origin'

os.makedirs(os.path.join(input_path, 'eval'), exist_ok = True)
eval_dir = os.path.join(input_path, 'eval')

os.makedirs(os.path.join(input_path, 'eval', 'source'), exist_ok = True)
eval_dir_source = os.path.join(input_path, 'eval', 'source')
os.makedirs(os.path.join(input_path, 'eval', 'target'), exist_ok = True)
eval_dir_traget = os.path.join(input_path, 'eval', 'target')

for i, folder in enumerate(tqdm(os.listdir(input_path))):
# for folder in os.listdir(raw_img):
    # print(folder)
    # print(os.path.join(str(eval_dir_source), str(folder) + '.png'))
    # move source
    for pic in os.listdir(os.path.join(input_path, folder)):
        number = pic[:9]
        back = pic[9:]
        if back == '_fake_des.png':
            real_img = number + '_des.png'
            shutil.copy(os.path.join(input_path, folder, pic), os.path.join(str(eval_dir_traget), pic))
            shutil.copy(os.path.join(input_path, folder, real_img), os.path.join(str(eval_dir_source), real_img))
            # print(number, back)
            # exit()