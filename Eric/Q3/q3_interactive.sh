#!/bin/bash
# Load required modules
MODULE="cuda"
ORIGIN="eifer4_q3_1"

if lsmod | grep "$MODULE" &> /dev/null ; then
    echo "$MODULE is already loaded"
else
    echo "Loading $MODULE"
    module load cuda/8.0-fasrc01

fi

# Pull data from git
git pull origin $ORIGIN

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

# push data back to git
git add -A
git commit -a -m "question 3 data added"
git config --global user.name "Eric Dunipace"
git config --global user.email edunipace@g.harvard.edu
git push origin $ORIGIN
