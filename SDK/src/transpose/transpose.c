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
 * Transpose:
 *     Transpose a fixed size(1024*1024) matrix which is stored in a two-dimension
 *     c array.
 *     Usage:
 *         "--thresh=<N>": The threshold used to check answer, default 0.01.
 */

#include "cuda2acc.h"
#include "timer.h"

#define M 1024
#define N 1024
float A[M][N], B[N][M], C[N][M];
#define elnum (M * N)
#define memsz (elnum * sizeof(float))

// Number of repetitions used for timing.
#define NUM_REPS 100

// transpose a matrix
void transposeCPU()
{
    // A(M, N) -> B(N, M)
    unsigned int i, j;
    for (i = 0; i < N; i++) {
        for (j = 0; j < M; j++) {
            B[i][j] = A[j][i];
        }
    }
}

// transpose a matrix with openacc
double transposeGPU()
{
    // A(M, N) -> C(N, M)
    unsigned int i, j;
    #pragma acc enter data copyin(A[0:elnum]) create(C[0:elnum])
    StartTimer();
    #pragma acc data present(A[0:elnum],C[0:elnum])
    {
    #pragma acc kernels loop independent
    for (i = 0; i < N; i++) {
        #pragma acc loop independent
        for (j = 0; j < M; j++) {
            C[i][j] = A[j][i];
        }
    }
    }
    double gettimer = GetTimer();
    #pragma acc exit data delete(A[0:elnum]) copyout(C[0:elnum])
    return gettimer;
}

// run test
void runtest(float th)
{
    finit_rand((float*)A, elnum);
    double tu_gpu = 0.0;

    for (int i = 0; i < NUM_REPS; i++) {
        transposeCPU();
    }

    // warm up to avoid timing startup
    transposeGPU();

    for (int i = 0; i < NUM_REPS; i++) {
        tu_gpu += transposeGPU(C, A);
    }

    float kernelBandwidth = 2.0f * 1000.0f * memsz/(1024*1024*1024)/(tu_gpu/NUM_REPS);
    printf("transpose GPU, Throughput = %.4f GB/s, Time = %f ms, Size = %u fp32 elements, NumDevsUsed = %u\n", kernelBandwidth, tu_gpu/NUM_REPS, M*N, 1);

    printf("%s\n", (fcheck((float*)B, (float*)C, elnum, th) ? "Test failed!" : "Test passed"));
}

// main function: process arguments and run tests
int main(int argc, char **argv)
{
char *names[] = { "thresh" };
    int flags[] = { 1 };
    int map[] = { 0 };
    struct OptionTable *opttable = make_opttable(1, names, flags, map);
    argproc(argc, argv, opttable);
    const char *str_th = opttable->table[0].val;

    float th = 0.01f;
    if (str_th)
        th = atof(str_th);

    printf("%s Starting...\n\n", argv[0]);
    print_gpuinfo(argc, (const char**)argv);
    printf("\nMatrix size: %dx%d\n",M,N);
    runtest(th);

    free_opttable(opttable);
    return 0;
}
