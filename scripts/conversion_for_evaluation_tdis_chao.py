import os
import shutil
from tqdm import tqdm


input_path = '/nfs/turbo/coe-ahowens/chfeng/tdis_results/'
output_num = 1
raw_img = os.path.join(input_path, 'tdis_sdedit_scale_7.0_emb_strength_0.75_uc_zero_False_strength_0.75')

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
    shutil.copy(os.path.join(str(raw_img), str(folder), 'reference_image.png'), os.path.join(str(eval_dir_source), str(folder) + '.png'))
    shutil.copy(os.path.join(str(raw_img), str(folder), 'reference_touch_image.png'), os.path.join(str(eval_dir_traget), str(folder) + '.png'))

    # move target
    output_len = output_num
    for i in range(output_len):
        image_name =  f"{i:05}.png"
        shutil.copy(os.path.join(str(raw_img), str(folder), image_name), os.path.join(str(eval_dir_generated), str(folder) + '_' + str(i) + '.png'))
    # exit()