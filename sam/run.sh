#!/bin/sh

#SBATCH -p spgpu
#SBATCH --account=fouhey2
#SBATCH --job-name=sam
#SBATCH --output=/nfs/turbo/fouheyTemp/dandans/handsv2/debug.log
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --open-mode=append

#SBATCH --gres=gpu:1
#SBATCH --time=1-00:00:00
#SBATCH --mem=45G



export LOGURU_LEVEL="INFO"
cd /nfs/turbo/fouheyTemp/dandans/handsv2/sam
conda activate hos



SPLIT=$1
INDEX=$2
echo SPLIT=$SPLIT
echo INDEX=$INDEX
set -x #echo ons


srun python run_sam_4handv2.py --split ${SPLIT} --index ${INDEX} 