/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

/*
 * This application demonstrates how to use the CUDA API to use multiple GPUs,
 * with an emphasis on simple illustration of the techniques (not on performance).
 *
 * Note that in order to detect multiple GPUs in your system you have to disable
 * SLI in the nvidia control panel. Otherwise only one GPU is visible to the
 * application. On the other side, you can still extend your desktop to screens
 * attached to both GPUs.
 */

#include <omp.h>
#include "cuda2acc.h"
#include "timer.h"

// DATA configuretion 
const int MAX_GPU_COUNT = 8;
const int N = 1048576 * 32;

float reduceKernel_GPU(float *h_Data, int N, int id, int GPU_N) {
    float sumGPU = 0;
    int chunk = N/GPU_N;
    int s = id*chunk;
    int t = (id+1)*chunk;
    #pragma acc kernels loop present(h_Data) reduction(+:sumGPU)
    for(int i = s; i < t; i++) {
        sumGPU += h_Data[i];
    }
    return sumGPU;
}

// Program main
int main()
{
    float *h_Data;
    float h_sumGPU[MAX_GPU_COUNT];
    double sumCPU, diff;

    printf("Starting simpleMultiGPU\n");

    int GPU_N = acc_get_num_devices(acc_device_nvidia);
    printf("CUDA-capable device count: %d\n",GPU_N);
    printf("Generating input data...\n\n");
    h_Data = (float *)malloc(N * sizeof (float));

    for (int i = 0; i < N; i++) {
        h_Data[i] = (float) rand() / (float)RAND_MAX;
    }

    printf("Computing with %d GPU's ... \n", GPU_N);
   
    StartTimer();

    #pragma omp parallel num_threads(GPU_N)
    {
        int id = omp_get_thread_num();
        acc_set_device_num(id+1, acc_device_nvidia);
        #pragma acc data copyin(h_Data[0:N])
        {
            h_sumGPU[id] = reduceKernel_GPU(h_Data, N, id, GPU_N);
        }
    }
    printf(" GPU Processing time : %f (ms) \n\n", GetTimer());

    printf("Computing with Host CPU... \n\n\n");

    float sumGPU = 0;

    for(int i = 0; i < GPU_N; i++)
    {
        printf("h_sumGPU[%d]=%f\n",i,h_sumGPU[i]);
        sumGPU += h_sumGPU[i];
    }

    sumCPU = 0;

    for(int i = 0; i < N; i++) {
        sumCPU += h_Data[i];
    }

    // Compare GPU and CPU results 
    printf("Comparing GPU and Host CPU results...\n");
    diff = fabs(sumCPU - sumGPU) / fabs(sumCPU);
    printf("  GPU sum: %f\n  CPU sum: %f\n", sumGPU, sumCPU);
    printf("  Relative difference: %E \n\n", diff);
    printf((diff < 1e-5) ? "Test PASSES\n\n" : "Test FAILS\n\n");

    // Cleanup and shutdown
    free(h_Data);
}
