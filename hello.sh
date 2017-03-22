export PGI=/n/seasfs03/IACS/cs205/pgi
export PATH=$PGI/linux86-64/16.10/bin:$PATH
export MANPATH=$MANPATH:$PGI/linux86-64/16.10/man
export LM_LICENSE_FILE=$LM_LICENSE_FILE:$PGI/license.dat

git pull
g++ helloacc.cpp -o helloacc_seq
pgc++ -acc helloacc.cpp -Minfo=accel -o helloacc_para
./helloacc
