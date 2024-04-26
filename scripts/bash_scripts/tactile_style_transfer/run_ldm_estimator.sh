export CUDA_VISIBLE_DEVICES=1
python scripts/tactile_style_transfer_estimator.py \
    --outdir outputs_nobackup/tactile_style_transfer_estimator_debug/ \
    --ddim_steps 200 \
    --config configs/tactile_style_transfer/gelslim2bubble.yaml \
    --ckpt logs/2024-01-18T16-19-17_gelslim2bubble/checkpoints/epoch=000029.ckpt \
    --max_sample -1 \
    --n_samples 10 \
    --scale 4