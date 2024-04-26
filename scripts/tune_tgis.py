import os

# ddim_step = [5,10,20,50,100,150,200]
strength = [0.5]
ddim_step = [200]
# strength = [0.5]

for step in ddim_step:
    for stre in strength:
        output_dir = str(step) + "_" + str(stre)
        os.system("python scripts/tdis.py --strength {} --ddim_steps {} --outdir outputs/tdis-final-supp-all/{}".format(str(stre), str(step), output_dir))