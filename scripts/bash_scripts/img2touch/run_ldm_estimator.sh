mode=$1
if [ $mode = "train" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python main.py --base configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbd.yaml -t --gpus 0,1,2,3
    # python main.py --base configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbd.yaml -t --gpus 0,1,2,3
    # python main.py --base configs/tactile_nerf/img2touch_tactile_nerf_resnet_db.yaml -t --gpus 0,1,2,3
    # python main.py --base configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbb.yaml -t --gpus 0,1,2,3
elif [ $mode = "eval" ]; then
    # python scripts/img2touch_tactile_nerf.py \
    #     --outdir outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_recalib/ \
    #     --ddim_steps 200 \
    #     --config configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbd.yaml \
    #     --ckpt logs/2023-11-25T08-33-19_img2touch_tactile_nerf_resnet_rgbd/checkpoints/epoch=000140.ckpt \
    #     --max_sample -1 \
    #     --n_samples 4 \
    #     --scale 7.5
    # python scripts/img2touch_tactile_nerf.py \
    #     --outdir outputs_nobackup/img2touch-tactile_nerf_dbg_interval/ \
    #     --ddim_steps 200 \
    #     --config configs/tactile_nerf/img2touch_tactile_nerf_resnet_db.yaml \
    #     --ckpt logs/2023-11-09T08-17-23_img2touch_tactile_nerf_resnet_db/checkpoints/epoch=000007.ckpt \
    #     --max_sample -1 \
    #     --n_samples 16 \
    #     --scale 7.5
    # RGBB
    # python scripts/img2touch_tactile_nerf.py \
    #     --outdir outputs_nobackup/img2touch-tactile_nerf_rgbbg_interval/ \
    #     --ddim_steps 200 \
    #     --config configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbb.yaml \
    #     --ckpt logs/img2touch_tactile_nerf_resnet_rgbb_final/checkpoints/epoch=000005.ckpt \
    #     --max_sample -1 \
    #     --n_samples 16 \
    #     --scale 7.5
    # # RDBDB without pretrain
    # python scripts/img2touch_tactile_nerf.py \
    #     --outdir outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_interpolated_train_random_rotation/ \
    #     --ddim_steps 200 \
    #     --config configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbd.yaml \
    #     --ckpt logs/img2touch_tactile_nerf_resnet_rgbdb_finetuned_final/checkpoints/epoch=000011.ckpt \
    #     --max_sample -1 \
    #     --n_samples 1 \
    #     --scale 7.5
    export CUDA_VISIBLE_DEVICES=1
    python scripts/img2touch_estimator_tactile_nerf.py \
        --outdir outputs_nobackup/img2touch-tactile_nerf_estimator_debug/ \
        --ddim_steps 200 \
        --config configs/tactile_nerf/img2touch_tactile_nerf_resnet_rgbd.yaml \
        --ckpt logs/img2touch_tactile_nerf_resnet_rgbdb_final_recalib/checkpoints/epoch=000012.ckpt \
        --max_sample -1 \
        --n_samples 4 \
        --scale 7.5
else
    echo "Invalid mode"
fi