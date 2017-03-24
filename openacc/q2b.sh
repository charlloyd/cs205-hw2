export PGI=/n/seasfs03/IACS/cs205/pgi
export PATH=$PGI/linux86-64/16.10/bin:$PATH
export MANPATH=$MANPATH:$PGI/linux86-64/16.10/man
export LM_LICENSE_FILE=$LM_LICENSE_FILE:$PGI/license.dat

#srun --mem-per-cpu=1000 -p holyseasgpu -n 1 --gres=gpu:1 --pty -t 0-01:00 /bin/bash
#module load gcc/6.3.0-fasrc01

git pull

#g++ q2b.cpp -o q2b_seq
pgc++ -acc q2b.cpp -Minfo=accel -o q2b_para
#./q2b_seq
./q2b_para
