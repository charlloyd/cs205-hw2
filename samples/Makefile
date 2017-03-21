
# 
#     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

#
# just typing 'make' prints out this help message
#

UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
help:
	@echo "To build and run a specific example, do the following:"
	@echo "    "
	@echo C Targets include:
	@echo "make acc_c1_test            : first simple example"
	@echo "make acc_c2_test            : second simple example"
	@echo "make acc_c2a_test           : second simple example, executed 3 times"
	@echo "make acc_c2i_test           : second simple example, with acc_init call"
	@echo "make acc_c3_test            : Jacobi relaxation"
	@echo "                              acc_c3 n m iters device"
	@echo "                              [defaults 1000 1000 10 nvidia]"
	@echo "make acc_c3a_test           : Jacobi relaxation with data region"
	@echo "make acc_c3i_test           : Jacobi relaxation without independent clause"
	@echo "make nbody_test             : nbody test(double precision)"
	@echo "make nbody_sp_test          : nbody test(single precision)"
	@echo "make nbody2_test            : nbody test with optimized data region(double precision)"
	@echo "make nbody2_sp_test         : nbody test with optimized data region(single precision)"
	@echo ""
	@echo Fortran Targets include:
	@echo "make acc_f1_test            : first simple example"
	@echo "make acc_f2_test            : second simple example"
	@echo "make acc_f2a_test           : second simple example, executed 3 times"
	@echo "make acc_f2i_test           : second simple example, with acc_init call"
	@echo "make acc_f3_test            : Jacobi relaxation"
	@echo "                              acc_f3 n m iters device"
	@echo "                              [defaults 1000 1000 10 nvidia]"
	@echo "make acc_f3a_test           : Jacobi relaxation with data region"
	@echo ""
	@echo SDK Targets include:
	@echo "make bilateralFilter_test   : bilateral Filter Test"
	@echo "make blackscholes_test      : blackscholes Test"
	@echo ""
	@echo "make all                    : all of the above tests"
else
help:
	@echo "To build and run a specific example, do the following:"
	@echo "    "
	@echo C Targets include:
	@echo "make acc_c1_test            : first simple example"
	@echo "make acc_c2_test            : second simple example"
	@echo "make acc_c2a_test           : second simple example, executed 3 times"
	@echo "make acc_c2i_test           : second simple example, with acc_init call"
	@echo "make acc_c3_test            : Jacobi relaxation"
	@echo "                              acc_c3 n m iters device"
	@echo "                              [defaults 1000 1000 10 nvidia]"
	@echo "make acc_c3a_test           : Jacobi relaxation with data region"
	@echo "make acc_c3i_test           : Jacobi relaxation without independent clause"
	@echo "make nbody_test             : nbody test(double precision)"
	@echo "make nbody_sp_test          : nbody test(single precision)"
	@echo "make nbody2_test            : nbody test with optimized data region(double precision)"
	@echo "make nbody2_sp_test         : nbody test with optimized data region(single precision)"
	@echo ""
	@echo Fortran Targets include:
	@echo "make acc_f1_test            : first simple example"
	@echo "make acc_f2_test            : second simple example"
	@echo "make acc_f2a_test           : second simple example, executed 3 times"
	@echo "make acc_f2i_test           : second simple example, with acc_init call"
	@echo "make acc_f3_test            : Jacobi relaxation"
	@echo "                              acc_f3 n m iters device"
	@echo "                              [defaults 1000 1000 10 nvidia]"
	@echo "make acc_f3a_test           : Jacobi relaxation with data region"
	@echo ""
	@echo "make all                    : all of the above tests"
endif

acc_c1_test:
	cd acc_c1; make build; make run; make clean

acc_c2_test:
	cd acc_c2; make build; make run; make clean

acc_c2a_test:
	cd acc_c2a; make build; make run; make clean

acc_c2i_test:
	cd acc_c2i; make build; make run; make clean

acc_c3_test:
	cd acc_c3; make build; make run; make clean

acc_c3a_test:
	cd acc_c3a; make build; make run; make clean

acc_c3i_test:
	cd acc_c3i; make build; make run; make clean

nbody_test:
	cd nbody; make PREC=FP64 build; make run; make clean

nbody_sp_test:
	cd nbody; make PREC=FP32 build; make run; make clean

nbody2_test:
	cd nbody2; make PREC=FP64 build; make run; make clean

nbody2_sp_test:
	cd nbody2; make PREC=FP32 build; make run; make clean


acc_f1_test:
	cd acc_f1; make build; make run; make clean

acc_f2_test:
	cd acc_f2; make build; make run; make clean

acc_f2a_test:
	cd acc_f2a; make build; make run; make clean

acc_f2i_test:
	cd acc_f2i; make build; make run; make clean

acc_f3_test:
	cd acc_f3; make build; make run; make clean

acc_f3a_test:
	cd acc_f3a; make build; make run; make clean

all: acc_c1_test acc_c2_test acc_c2a_test acc_c2i_test acc_c3_test    \
     acc_c3a_test acc_c3i_test nbody_test nbody_sp_test nbody2_test   \
     nbody2_sp_test acc_f1_test acc_f2_test acc_f2a_test acc_f2i_test \
     acc_f3_test acc_f3a_test
