#!/bin/bash
########## SBATCH Lines for Resource Request ##########
 
#SBATCH --time=01:30:00             # limit of wall clock time - how long the job will run (same as -t)
#SBATCH --nodes=1                   # number of different nodes - could be an exact number or a range of nodes (same as -N)
#SBATCH --ntasks=1                  # number of tasks - how many tasks (nodes) that you require (same as -n)
#SBATCH --cpus-per-task=1           # number of CPUs (or cores) per task (same as -c)
#SBATCH --mem-per-cpu=8G            # memory required per allocated CPU (or core) - amount of memory (in bytes)
#SBATCH --job-name standard_softplus    # you can give your job a name for easier identification (same as -J)
#SBATCH -o "runinfo_standard_softplus.out"
#SBATCH --constraint="amd20"

module load FFmpeg

python main.py -I standard_test -O standard_softplus_arch2 --train_iters=3 --n_episodes=10 --dprobe=5 --activation=softplus --savefreq=50 -N -P

cd standard_softplus_arch2

ffmpeg -r 10 -f image2 -i slime_test_%04d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p standard_softplus_arch2.mp4
