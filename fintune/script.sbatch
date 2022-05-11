#!/bin/bash
#
#SBATCH --job-name=project
#SBATCH --output=project.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mail-type=END
#SBATCH --mail-user=yz7043@nyu.edu

module purge;
module load anaconda3/2020.07;
source /share/apps/anaconda3/2020.07/etc/profile.d/conda.sh;
conda activate /scratch/yz7043/envs_dirs/torch37;
export PATH=/scratch/yz7043/envs_dirs/torch37/bin:$PATH;
cd /scratch/yz7043/project;
python main.py --saveImage True --train True --epochs 10 --modelPath test_next.pt;
