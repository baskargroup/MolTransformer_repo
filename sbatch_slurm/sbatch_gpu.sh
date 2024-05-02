#!/bin/bash
#SBATCH --time=24:00:00   # walltime limit (HH:MM:SS)
#SBATCH --nodes=1 # number of nodes
#SBATCH --exclusive
#SBATCH --ntasks-per-node=16 # 16 processor core(s) per node
#SBATCH --mem=369G   # maximum memory per node
#SBATCH --gres=gpu:a100:4
#SBATCH -A baskargroup-a100gpu
#SBATCH --job-name="ddp gpu"
#SBATCH --mail-user=chyang@iastate.edu   # email address
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
# LOAD MODULES, INSERT CODE, AND RUN YOUR PROGRAMS HERE

module load miniconda3
source activate /work/mech-ai/bella/selfies_envs
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --nproc_per_node=4 /work/mech-ai/bella/MolTransformer/main.py  