from moviepy.editor import ImageSequenceClip, clips_array
import os
import json

seq2shuffled = {}
split_shuffled = json.load(open('../tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/split_interval.json'))['test']
split_seq = json.load(open('../tactile_nerf/vision_touch_pairs_tactile_nerf_final_recalib/split_interval_vis.json'))['test']
for iter,inst in enumerate(split_seq):
    seq2shuffled[iter]= f'{split_shuffled.index(inst)}'.zfill(4) + '.png'


# Set the paths to your folders with images
folder1_path = '../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_recalib/eval/input'
folder2_path = '../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_recalib/eval/pred'
folder3_path = '../outputs_nobackup/img2touch-tactile_nerf_rgbdbg_interval_recalib/eval/gt'

# Load the images from each folder
images1 = [os.path.join(folder1_path, seq2shuffled[i]) for i in range(len(sorted(os.listdir(folder1_path)))) ]
images2 = [os.path.join(folder2_path, seq2shuffled[i]) for i in range(len(sorted(os.listdir(folder2_path)))) ]
images3 = [os.path.join(folder3_path, seq2shuffled[i]) for i in range(len(sorted(os.listdir(folder3_path)))) ]

# Make sure all lists have the same number of images
assert len(images1) == len(images2) == len(images3), "The folders don't contain the same number of images."

# Set fps (frames per second) to your desired value
fps = 10

# Create clips for each image sequence
clip1 = ImageSequenceClip(images1, fps=fps)
clip2 = ImageSequenceClip(images2, fps=fps)
clip3 = ImageSequenceClip(images3, fps=fps)

# Resize clips to a third of the width of the largest clip to fit them side by side
max_width = max(clip1.w, clip2.w, clip3.w)
clip1_resized = clip1.resize(width=400)
clip2_resized = clip2.resize(width=400)
clip3_resized = clip3.resize(width=400)

# Combine the three clips side by side
final_clip = clips_array([[clip1_resized, clip2_resized, clip3_resized]])

# Write the final video to a file
final_clip.write_videofile("three_split_screen_video.mp4", codec='libx264')
