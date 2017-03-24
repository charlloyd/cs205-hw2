/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/time.h>
#endif

#include <omp.h>
#include <math.h>
#include <float.h>
#include "openacc.h"

#ifdef __cplusplus
#include "timer.h"
extern "C" {
#endif

typedef unsigned char uchar;
#if defined (__APPLE__) || defined(MACOSX)
typedef unsigned int uint;
#endif

extern void finit_rand(float *vec, uint32_t len);
extern void iinit_rand(int *vec, uint32_t len);
extern void dinit_rand(double *vec, uint32_t len);
extern int32_t fcheck(float *A, float *B, uint32_t N, float th);
extern int32_t dcheck(double *A, double *B, uint32_t N, double th);
extern int32_t icheck(int *A, int *B, uint32_t N, int th);
extern float calc_bandwidth(int32_t memsize, double time);
extern void dump(float *arr, int size);
extern int fread_float(const char *fn, float *data, int count);
extern int fread_int(const char *fn, int *data, int count);

#define ABS(A) ((A) > 0 ? (A) : -(A))
#define MIN(A, B) ((A) > (B) ? (B) : (A))
#define MAX(A, B) ((A) < (B) ? (B) : (A))

#ifdef __cplusplus
}
#endif

#endif
