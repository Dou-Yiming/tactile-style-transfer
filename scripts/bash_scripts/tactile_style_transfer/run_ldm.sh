mode=$1
if [ $mode = "train" ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python main.py --base configs/tactile_style_transfer/gelslim2bubble.yaml -t --gpus 0,1,2,3

elif [ $mode = "eval" ]; then
    export CUDA_VISIBLE_DEVICES=0
    python gelslim2bubble.py \
        --outdir outputs/gelslim2bubble_new_test/ \
        --ddim_steps 200 \
        --config configs/tactile_style_transfer/gelslim2bubble.yaml \
        --ckpt logs/2024-03-01T00-24-07_gelslim2bubble/checkpoints/epoch=000029.ckpt \
        --max_sample -1 \
        --n_samples 10 \
        --scale 4
else
    echo "Invalid mode"
fi