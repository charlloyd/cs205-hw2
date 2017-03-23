#!/bin/bash
# Load required modules
MODULE="cuda"

if lsmod | grep "$MODULE" &> /dev/null ; then

else
    module load cuda/8.0-fasrc01

fi

# Pull data from git
git pull origin eifer4_q3_1

# Make appropriate files
cd
make -C cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_naive
make -C cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_shmem

# Run target files
cd cs205-hw2/CleanGPU/exercise_solutions/cuda/matmul_GPU_naive
./x.matmul_GPU_naive

cd ..
cd matmul_GPU_shmem
./x.matmul_GPU_shmem
