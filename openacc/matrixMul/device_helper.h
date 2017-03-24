/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef DEVICE_H
#define DEVICE_H

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "cuda_device_runtime_api.h"

#ifdef __cplusplus
#include "helper_functions.h"
#include "helper_cuda.h"
#endif

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64) && defined(_MSC_VER)
typedef unsigned __int64 memsize_t;
#else
#include <stdint.h>
typedef uint64_t memsize_t;
#endif

#ifdef __cplusplus
extern "C" {
#endif

extern void print_gpuinfo(int, const char **);
bool getTargetDeviceGlobalMemSize(memsize_t *, const int, const char **);

#ifdef __cplusplus
}
#endif

#endif
