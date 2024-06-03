#!/bin/bash
#SBATCH --job-name=EDSR          # create a short name for your job
#SBATCH --partition=ais-gpu 
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --gpus=8
#SBATCH --cpus-per-task=64       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=8G
#SBATCH --time=6-00:00:00        # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email when job fails
#SBATCH --mail-user=m.aleshin@skoltech.ru
#SBATCH --output=/trinity/home/m.aleshin/projects/superresolution/EDiffSR/experiments/sisr/slurm_logs/ediffsr_%x_%j.txt   

source /beegfs/home/m.aleshin/.bashrc
conda activate torch

cd /beegfs/home/m.aleshin/projects/superresolution/EDiffSR/codes/config/sisr

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=13110 \
    --use_env test.py -opt options/test/setting.yml \
    --launcher pytorch