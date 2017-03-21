/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <timer.h>
#include "cuda2acc.h"
#define totalTheads  16 * 1024 * 1024

void increment_kernelAcc(int *g_data, int inc_value)
{
    #pragma acc kernels loop deviceptr(g_data)
    for (int i = 0; i < totalTheads; i++) {
        g_data[i] = g_data[i] + inc_value;	 
    }
}

int correct_output(int *data, const int n, const int x)
{
    for (int i = 0; i < n; i++) {
        if (data[i] != x) {
            return 0;
        }
    }
    return 1;
}

int main(int argc ,  char **argv)
{
    int n = 16 * 1024 * 1024;
    int nbytes = n * sizeof (int);
    int value = 26;
    int *a = NULL;

    printf("[asyncAPI]\n");
    print_gpuinfo(argc, (const char **)argv);

    a = (int *)malloc(nbytes);
    memset(a, 0, nbytes);

    //allocate device memory
    int *d_a = NULL;
    d_a = (int *)acc_malloc(nbytes);
  
    #pragma acc kernels loop deviceptr(d_a)
    for (int i = 0; i < n; i++) {
        d_a[i] = 255;
    }

    StartTimer();
    #pragma acc data async	
    acc_memcpy_to_device(d_a, a, nbytes);
    #pragma acc wait
    increment_kernelAcc(d_a, value);
    #pragma acc wait
    #pragma acc data  async(n - 1)
    acc_memcpy_from_device(a, d_a, nbytes);
    double gpu_time = GetTimer();
    printf("time spent executing by the GPU: %.2fms\n", gpu_time);

    printf("--------------------------------------------------------------\n");
    printf("[asyncAPI] -> Test Results:\n");
    if( correct_output(a, n, value) ) {
        printf("Test PASSES\n");
    } else {
        printf("Test FAILS\n");
    }
    return 0;
}
