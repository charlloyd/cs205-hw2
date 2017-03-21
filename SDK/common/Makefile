#
#     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

# SOURCE FILES
UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
STATIC_LIB := libcommon.lib
else
STATIC_LIB := libcommon.a
endif

SRCFILES = common.c argproc.c device_helper.cpp
CUDAFLAGS = -Mcuda
ROOTDIR := .

include common.mk
