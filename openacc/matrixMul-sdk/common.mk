# 
#     Copyright (c) 2013, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 

################################################################################
#
# Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
#
# NVIDIA Corporation and its licensors retain all intellectual property and 
# proprietary rights in and to this software and related documentation. 
# Any use, reproduction, disclosure, or distribution of this software 
# and related documentation without an express license agreement from
# NVIDIA Corporation is strictly prohibited.
#
# Please refer to the applicable NVIDIA end user license agreement (EULA) 
# associated with this source code for terms and conditions that govern 
# your use of this NVIDIA software.
#
################################################################################
#
# Common build script for CUDA source projects for Linux and Mac platforms
#
################################################################################
#
#          THIS CODE AND INFORMATION ARE PROVIDED "AS IS" WITHOUT
#   WARRANTY OF ANY KIND, EITHER EXPRESSED OR IMPLIED, INCLUDING BUT
#   NOT LIMITED TO THE IMPLIED WARRANTIES OF MERCHANTABILITY AND/OR
#   FITNESS FOR A PARTICULAR PURPOSE.
#
################################################################################
#

UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
  OBJ=obj
  EXESUFFIX=.exe
else
  OBJ=o
endif

.SUFFIXES : .cu .c .cpp

# Add new SM Versions here as devices with new Compute Capability are released
SM_VERSIONS   := 10 11 12 13 20

# detect OS
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = 

# detect 32-bit or 64-bit platform
HP_64 = $(shell uname -m | grep 64)
OSARCH= $(shell uname -m)

# Basic directory setup for SDK
# (override directories only if they are not already defined)
SRCDIR     ?= ./
ROOTDIR    ?= ..
ROOTBINDIR ?= $(ROOTDIR)/../bin
BINDIR     ?= $(ROOTBINDIR)/$(OSLOWER)
ROOTOBJDIR ?= obj
LIBDIR     := $(ROOTDIR)/../lib
COMMONDIR  := $(ROOTDIR)/../common
INCDIR     := $(ROOTDIR)/../include

# Compilers
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
CXX      := pgcpp
CC         := pgcc
else
CXX      := pgc++
CC         := pgcc
endif
LINK       := $(CXX)

# Includes
INCLUDES  += -I. -I$(INCDIR)

# Warning flags
CXXWARN_FLAGS =
CWARN_FLAGS := $(CXXWARN_FLAGS)

# architecture flag for compilers build
CXX_ARCH_FLAGS  :=
LIB_ARCH        := $(OSARCH)

# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(x86_64),1) 
    LIB_ARCH         = x86_64
    CXX_ARCH_FLAGS += -m64
else 
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(i386),1)
        LIB_ARCH         = i386
        CXX_ARCH_FLAGS += -m32
    else 
        LIB_ARCH        = x86_64
        CXX_ARCH_FLAGS += -m64
    endif
endif

# Debug/release configuration
ifeq ($(dbg),1)
        COMMONFLAGS += -g
        BINSUBDIR   := debug
        LIBSUFFIX   :=
else
        COMMONFLAGS += -acc -Minfo=accel $(CUDAFLAGS) $(EXTRAFLAGS)
        BINSUBDIR   := release
        LIBSUFFIX   :=
endif

# Libs
LIB       := -L$(LIBDIR)

CFLAGS    += $(CWARN_FLAGS) $(CXX_ARCH_FLAGS) $(OPT)
CXXFLAGS  += $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS) $(OPT)
PGCUFLAGS += $(CXXFLAGS)
LINKFLAGS += -pgf90libs $(COMMONFLAGS) $(OPT) $(LIB)
LINK      += $(LINKFLAGS) $(CXX_ARCH_FLAGS)

# Common flags
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
COMMONFLAGS += $(INCLUDES)
else
COMMONFLAGS += $(INCLUDES)
endif

ifeq ($(USECUFFT),1)
  ifeq ($(emu),1)
    LIB += -lcufftemu
  else
    LIB += -lcufft
  endif
endif

ifeq ($(USECUBLAS),1)
  LIB += -lcublasemu -llapack -lblas -pgf90libs
endif

# Lib/exe configuration
ifneq ($(STATIC_LIB),)
TARGETDIR := $(LIBDIR)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
TARGET   := $(subst .lib,_$(LIB_ARCH)$(LIBSUFFIX).lib,$(LIBDIR)/$(STATIC_LIB))
else
TARGET   := $(subst .a,_$(LIB_ARCH)$(LIBSUFFIX).a,$(LIBDIR)/$(STATIC_LIB))
endif
LINKLINE  = ar rucv $(TARGET) $(OBJS)
else
ifneq ($(OMIT_CUTIL_LIB),1)
LIB += -lcommon_$(LIB_ARCH)$(LIBSUFFIX)
endif
# Device emulation configuration
ifeq ($(emu), 1)
# consistency, makes developing easier
CXXFLAGS		+= -D__DEVICE_EMULATION__
CFLAGS			+= -D__DEVICE_EMULATION__
endif
TARGETDIR := $(BINDIR)/$(BINSUBDIR)
TARGET    := $(TARGETDIR)/$(EXECUTABLE)
LINKLINE  = $(LINK) -o $(TARGET) $(OBJS) $(LIB)
endif

# check if verbose 
ifeq ($(verbose), 1)
	VERBOSE :=
else
	VERBOSE := 
endif

################################################################################
# Check for input flags and set compiler flags appropriately
################################################################################
ifeq ($(fastmath), 1)
	COMMONFLAGS +=
endif

ifeq ($(verbose), 1)
        COMMONFLAGS += -v
endif

# Add common flags
PGCUFLAGS += $(COMMONFLAGS)
CXXFLAGS  += $(COMMONFLAGS)
CFLAGS    += $(COMMONFLAGS)

################################################################################
# Set up source fiels
################################################################################
CFILES = $(filter %.c, $(SRCFILES))
CPPFILES = $(filter %.cpp, $(SRCFILES))
CUDAFILES = $(filter %.cu, $(SRCFILES))

################################################################################
# Set up object files
################################################################################

OBJDIR := $(ROOTOBJDIR)/$(LIB_ARCH)/$(BINSUBDIR)
OBJS +=  $(patsubst %.cpp,$(OBJDIR)/%.cpp.$(OBJ),$(notdir $(CPPFILES)))
OBJS +=  $(patsubst %.c,$(OBJDIR)/%.c.$(OBJ),$(notdir $(CFILES)))
OBJS +=  $(patsubst %.cu,$(OBJDIR)/%.cu.$(OBJ),$(notdir $(CUDAFILES)))

################################################################################
# Rules
################################################################################
$(OBJDIR)/%.c.$(OBJ) : $(SRCDIR)%.c $(C_DEPS)
	$(VERBOSE)$(CC) $(CFLAGS) -o $@ -c $<

$(OBJDIR)/%.cpp.$(OBJ) : $(SRCDIR)%.cpp $(C_DEPS)
	$(VERBOSE)$(CXX) $(CXXFLAGS) $(CXXMPFLAGS) -o $@ -c $<

# Default arch includes gencode for sm_10, sm_20, and other archs from GENCODE_ARCH declared in the makefile
$(OBJDIR)/%.cu.$(OBJ) : $(SRCDIR)%.cu $(CU_DEPS)
	$(VERBOSE)$(CXX) $(PGCUFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

#
# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding 
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
# This will be used over the default rule abov
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#

# This line invokes the above template for each arch version stored in
# SM_VERSIONS.  The call funtion invokes the template, and the eval
# function interprets it as make commands.
$(foreach smver,$(SM_VERSIONS),$(eval $(call SMVERSION_template,$(smver))))

$(TARGET): makedirectories $(OBJS) Makefile
	$(VERBOSE)$(LINKLINE)

makedirectories:
	$(VERBOSE)mkdir -p $(LIBDIR)
	$(VERBOSE)mkdir -p $(OBJDIR)
	$(VERBOSE)mkdir -p $(TARGETDIR)

tidy :
	$(VERBOSE)find . | egrep "#" | xargs rm -f
	$(VERBOSE)find . | egrep "\~" | xargs rm -f

clean : tidy
	$(VERBOSE)rm -f $(OBJS)
	$(VERBOSE)rm -f $(TARGET)
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.ppm
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.pgm
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bin
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.bmp
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.pdb
	$(VERBOSE)rm -f $(ROOTBINDIR)/$(OSLOWER)/$(BINSUBDIR)/*.dwf

clobber : clean
	$(VERBOSE)rm -rf $(ROOTOBJDIR)
