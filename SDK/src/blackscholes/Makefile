#
#     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

CC       = pgcc
FC       = pgfortran
OBJ=o
EXE=out
RUN=
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
   OBJ=obj
   EXE=exe
endif

ACCFLAGS = $(OPT) -fast -Minline=levels:2 -acc -Minfo=acc -I../../include
ACCFLAGS2 = $(OPT) -fast -acc -Minfo=acc -I../../include
OMPFLAGS = -fast -Minline -mp=allcores -Minfo=par -I../../include

all: build run verify

build: blackscholes.c blackscholes.F90 blackscholes_routine.c
	$(CC) $(ACCFLAGS) -o blackscholesc_acc.$(EXE) blackscholes.c
	$(CC) $(OMPFLAGS) -o blackscholesc_omp.$(EXE) blackscholes.c
	$(FC) $(ACCFLAGS) -o blackscholesf_acc.$(EXE) blackscholes.F90
	$(FC) $(OMPFLAGS) -o blackscholesf_omp.$(EXE) blackscholes.F90
	$(CC) $(ACCFLAGS2) -o blackscholesc_acc_routine.$(EXE) blackscholes_routine.c


run: blackscholesc_acc.$(EXE) blackscholesc_omp.$(EXE) blackscholesf_acc.$(EXE) blackscholesf_omp.$(EXE) blackscholesc_acc_routine.$(EXE)
	@echo "------------------------- blackscholesc_acc.$(EXE) -----------------------"
	$(RUN) ./blackscholesc_acc.$(EXE)
	@echo "------------------------- blackscholesc_omp.$(EXE) -----------------------"
	$(RUN) ./blackscholesc_omp.$(EXE)
	@echo "------------------------- blackscholesf_acc.$(EXE) -----------------------"
	$(RUN) ./blackscholesf_acc.$(EXE)
	@echo "------------------------- blackscholesf_omp.$(EXE) -----------------------"
	$(RUN) ./blackscholesf_omp.$(EXE)
	@echo "------------------------- blackscholesc_acc_routine.$(EXE) ---------------"
	$(RUN) ./blackscholesc_acc_routine.$(EXE)

verify:


clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.oo *.mod *.dwf *.pdb prof
