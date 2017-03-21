#
#     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

EXECUTABLE = simpleCUBLAS$(EXESUFFIX)
SRCFILES = simpleCUBLAS.cpp
CUDAFLAGS = -Mcuda -Mcudalib=cublas
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
EXTRAFLAGS = -defaultlib:cublas
else
EXTRAFLAGS = -lcublas
endif

include ../../common/common.mk
