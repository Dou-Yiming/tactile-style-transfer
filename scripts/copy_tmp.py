import os
import shutil

insts = [2,3,9,33,121,215]
for inst in insts:
    input_path = f'../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_recalib/eval/input/{inst:04}.png'
    gt_path = f'../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/gt/{inst:04}.png'
    norgb_path = f'../outputs_nobackup/img2touch-tactile_nerf_dbg_interval/eval/pred/{inst:04}.png'
    nodepth_path = f'../outputs_nobackup/img2touch-tactile_nerf_rgbbg_interval/eval/pred/{inst:04}.png'
    nopretrain_path = f'../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_nopretrain/eval/pred/{inst:04}.png'
    noreranking_path = f'../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval_argmin/pred/{inst:04}.png'
    full_path = f'../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_new/eval/pred/{inst:04}.png'
    
    shutil.copy(input_path, f'./tmp_vis/{inst:04}_input.png')
    shutil.copy(gt_path, f'./tmp_vis/{inst:04}_gt.png')
    shutil.copy(norgb_path, f'./tmp_vis/{inst:04}_norgb.png')
    shutil.copy(nodepth_path, f'./tmp_vis/{inst:04}_nodepth.png')
    shutil.copy(nopretrain_path, f'./tmp_vis/{inst:04}_nopretrain.png')
    shutil.copy(noreranking_path, f'./tmp_vis/{inst:04}_noreranking.png')
    shutil.copy(full_path, f'./tmp_vis/{inst:04}_full.png')
    