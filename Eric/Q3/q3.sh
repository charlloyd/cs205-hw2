#!/bin/bash
#SBATCH -J CUDA_MAT_MUL
#SBATCH -o cuda_mm.out
#SBATCH -e cuda_mm.err
#SBATCH -p holyseasgpu
#SBATCH -n 1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000

# Load required modules
module load cuda/8.0-fasrc01

# Pull data from git
git pull origin master

# Make appropriate files
make -C cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_naive
make -C cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_shmem

# Run target files
cd cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_naive
./x.matmul_GPU_naive

cd cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_shmem
./x.matmul_GPU_shmem
