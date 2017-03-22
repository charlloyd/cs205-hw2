export PGI=/n/seasfs03/IACS/cs205/pgi
export PATH=$PGI/linux86-64/16.10/bin:$PATH
export MANPATH=$MANPATH:$PGI/linux86-64/16.10/man
export LM_LICENSE_FILE=$LM_LICENSE_FILE:$PGI/license.dat

#module load gcc/6.3.0-fasrc01

git pull
g++ q2_b.cpp -o q2_b_seq
pgc++ -acc q2_b.cpp -Minfo=accel -o q2_b_para
./q2_b_seq
./q2_b_para
