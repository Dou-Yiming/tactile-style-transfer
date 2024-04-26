import os
import shutil
from tqdm import tqdm


input_path = '/home/fredyang/fredyang/stable-diffusion/outputs/tdis_final_save_full/200_0.75'
output_num = 4
raw_img = os.path.join(input_path, 'full_images')

os.makedirs(os.path.join(input_path, 'eval'), exist_ok = True)
eval_dir = os.path.join(input_path, 'eval')

os.makedirs(os.path.join(input_path, 'eval', 'A'), exist_ok = True)
eval_dir_source = os.path.join(input_path, 'eval', 'A')
os.makedirs(os.path.join(input_path, 'eval', 'B'), exist_ok = True)
eval_dir_traget = os.path.join(input_path, 'eval', 'B')
os.makedirs(os.path.join(input_path, 'eval', 'G'), exist_ok = True)
eval_dir_generated = os.path.join(input_path, 'eval', 'G')

for i, folder in enumerate(tqdm(os.listdir(raw_img))):
# for folder in os.listdir(raw_img):
    # print(folder)
    # print(os.path.join(str(eval_dir_source), str(folder) + '.png'))
    # move source
    shutil.copy(os.path.join(str(raw_img), str(folder), 'imageA.png'), os.path.join(str(eval_dir_source), str(folder) + '.png'))
    shutil.copy(os.path.join(str(raw_img), str(folder), 'imageB.png'), os.path.join(str(eval_dir_traget), str(folder) + '.png'))

    # move target
    output_len = output_num
    for i in range(output_len):
        image_name =  f"{i:05}.png"
        shutil.copy(os.path.join(str(raw_img), str(folder), image_name), os.path.join(str(eval_dir_generated), str(folder) + '_' + str(i) + '.png'))
    # exit()