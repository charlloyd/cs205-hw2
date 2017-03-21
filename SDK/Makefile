#
#     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# 


UNAME := $(shell uname -a)
ifeq ($(findstring CYGWIN_NT, $(UNAME)), CYGWIN_NT)
SOURCES = \
        alignedTypes \
        bilateralFilter \
        blackscholes \
        FDTD3d \
        histogram \
        seismic_cpml \
        simpleTemplates \
        scalarProd \
        template \
        transpose \
        transposeNew \
        vectorAdd \
        asyncAPI \
        bandwidthTest \
        cppIntegration \
        deviceQuery \
        matrixMul \
        MonteCarloMultiGPU \
        quasirandomGenerator \
        reduction \
        simpleMultiGPU
else
SOURCES = \
	alignedTypes \
	bilateralFilter \
	blackscholes \
	FDTD3d \
	histogram \
	seismic_cpml \
	simpleTemplates \
	scalarProd \
	template \
	transpose \
	transposeNew \
	vectorAdd \
	asyncAPI \
	bandwidthTest \
	cppIntegration \
	deviceQuery \
	matrixMul \
	MonteCarloMultiGPU \
	quasirandomGenerator \
	reduction \
	simpleCUBLAS \
	simpleMultiGPU
endif

STATIC_LIB := libcommon.a

all: lib src

lib:
	@echo Building common ...
	make -C common

$(SOURCES):
	make -C src/$@

src:
	@for i in $(SOURCES); do \
		make -C src/$${i} ; \
	done

clean_lib:
	make -C common clean

clean_src:
	@for i in $(SOURCES); do \
		make -C src/$${i} clean; \
	done

clean: clean_src clean_lib

.PHONY: clean clean_src clean_lib lib src all
