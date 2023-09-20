#!/bin/bash
#SBATCH --account soc-gpu-np
#SBATCH --partition soc-gpu-np
#SBATCH --ntasks-per-node=32
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --mem=24GB
#SBATCH --mail-user=u1468310@utah.edu
#SBATCH --mail-type=FAIL,END
#SBATCH -o assignment_1-%j
#SBATCH --export=ALL
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mp1
OUT_DIR=/scratch/general/vast/u1468310/cs6957/assignment1/models
mkdir -p ${OUT_DIR}
python mp1.py --output_dir ${OUT_DIR} <other arguments>