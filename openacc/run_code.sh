export PGI=/n/seasfs03/IACS/cs205/pgi
export PATH=$PGI/linux86-64/16.10/bin:$PATH
export MANPATH=$MANPATH:$PGI/linux86-64/16.10/man
export LM_LICENSE_FILE=$LM_LICENSE_FILE:$PGI/license.dat

ORIGIN = "q3_test"

git pull $ORIGIN

make all_seq
make all_para
