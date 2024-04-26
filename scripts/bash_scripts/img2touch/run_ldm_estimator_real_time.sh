export CUDA_VISIBLE_DEVICES=1
python scripts/img2touch_estimator_tactile_nerf_real_time.py \
    --outdir outputs/tarf_estimator_real_time/ \
    --ddim_steps 200 \
    --config configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbd.yaml \
    --ckpt logs/img2touch_tactile_nerf_resnet_rgbdb_final_recalib/checkpoints/epoch=000012.ckpt \
    --max_sample -1 \
    --n_samples 4 \
    --scale 7.5