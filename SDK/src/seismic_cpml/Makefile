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

ACCFLAGS = -fast -acc -Minfo=acc $(OPT)

all: build run verify

build: seismic_PML_Collino_3D_isotropic.F90
	$(FC) $(ACCFLAGS) -o seismic_cmpl.$(EXE) seismic_PML_Collino_3D_isotropic.F90

run: seismic_cmpl.$(EXE)
	$(RUN) ./seismic_cmpl.$(EXE) 32 32 32 2 150 100 >seismic_out.log  2>seismic_out.err


verify: check.F
	@pgf90 -o check.$(EXE) check.F
	./check.$(EXE)

clean:
	@echo 'Cleaning up...'
	@rm -rf *.$(EXE) *.$(OBJ) *.oo *.mod *.dwf *.pdb prof *.pnm *.dat timestamp* plot* seismic_out.*
