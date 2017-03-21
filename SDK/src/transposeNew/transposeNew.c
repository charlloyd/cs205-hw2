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
 * Transpose New:
 *     Transpose a matrix which is stored in a one-dimension array and in
 *         a C 2-dimension (float**) array.
 *     Usage:
 *         "--m=<N>": Specify width of the matrix transposed, default 128.
 *         "--n=<N>": Specify height of the matrix transposed, default 128.
 *         "--reps=<N>": Specify number of repetitions, default 100.
 *         "--thresh=<N>": The threshold used to check answer, default 0.00.  This is ignored.
 */

#include "cuda2acc.h"
#include "timer.h"

void transposeCPU(float *A, float *B, int N, int M)
{
    // A(M, N) -> B(N, M)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

// 2d version
void transposeCPU2d(float **A, float **B, int N, int M)
{
    // A(M, N) -> B(N, M)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            B[j][i] = A[i][j];
        }
    }
}

// transpose a matrix (implemented in one-dimensional C array) using openacc
// return collected time
double transposeGPU(float *A, float *B, int N, int M)
{
    // A(M, N) -> C(N, M)
    int elnum = N*M;
    #pragma acc enter data copyin(A[0:elnum]) create(B[0:elnum])
    StartTimer();
    // openacc 1.0 doesn't support dynamically allocated two-dimensional array
    #pragma acc kernels loop independent present(A[0:elnum],B[0:elnum])
    for (int i = 0; i < M; i++) {
        #pragma acc loop independent
        for (int j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
    double gettimer = GetTimer();
    #pragma acc exit data delete(A[0:elnum]) copyout(B[0:elnum])
    return gettimer;
}

// dynamically allocated two-dimensional array version using openacc 2.0
double transposeGPU2d(float **A, float **B, int N, int M)
{
    // A(M, N) -> B(N, M)
    #pragma acc enter data copyin(A[0:M][0:N]) create(B[0:N][0:M])
    StartTimer();
    // do work
    #pragma acc parallel loop present(A[0:M][0:N],B[0:N][0:M])
    for (int i = 0; i < M; i++) {
        #pragma acc loop
        for (int j = 0; j < N; j++) {
            B[j][i] = A[i][j];
        }
    }
    double gettimer = GetTimer();
    #pragma acc exit data delete(A[0:M][0:N]) copyout(B[0:N][0:M])
    return gettimer;
}

// allocate dynamic 2d array and do test
void runtest2d(float th, int N, int M, int NUM_REPS)
{
    // init inputs
    float **A, **B, **C;
    int elnum = N*M;
    int memsz = elnum * sizeof(float);
    double tu_gpu = 0.0;

    A = (float **)malloc(sizeof(float) * elnum + sizeof(float *) * M);
    for (int i = 0; i < M; i++)
        A[i] = (float *)(A + M) + i * N;
    B = (float **)malloc(sizeof(float) * elnum + sizeof(float *) * N);
    C = (float **)malloc(sizeof(float) * elnum + sizeof(float *) * N);
    for (int i = 0; i < N; i++) {
        B[i] = (float *)(B + N) + i * M;
        C[i] = (float *)(C + N) + i * M;
    }
    finit_rand((float*)(A + M), elnum);

    // do work
    transposeCPU2d(A, B, N, M);

    // warm up the GPU
    transposeGPU2d(A, C, N, M);

    for (int i = 0; i < NUM_REPS; i++) {
        tu_gpu += transposeGPU2d(A, C, N, M);
    }

    float kernelBandwidth = 2.0f * 1000.0f * memsz/(1024*1024*1024)/(tu_gpu/NUM_REPS);
    printf("transpose GPU 2d, Throughput = %.4f GB/s, Time = %f ms, Size = %u fp32 elements, NumDevsUsed = %u\n", kernelBandwidth, tu_gpu/NUM_REPS, M*N, 1);

    printf("%s\n", (fcheck((float *)(B + N), (float *)(C + N), elnum, th) ? "Test FAILED!" : "Test PASSED"));



    // free space
    free(A);
    free(B);
    free(C);
}

// allocate memory and do test
void runtest(float th, int N, int M, int NUM_REPS)
{
    float *A, *B, *C;
    int elnum = N*M;
    int memsz = elnum * sizeof(float);
    double tu_cpu = 0.0, tu_gpu = 0.0;

    A = (float*)malloc(sizeof(float) * elnum);
    B = (float*)malloc(sizeof(float) * elnum);
    C = (float*)malloc(sizeof(float) * elnum);
    finit_rand((float*)A, elnum);

    transposeCPU(A, B, N, M);

    // warm up the GPU
    transposeGPU(A, C, N, M);

    for (int i = 0; i < NUM_REPS; i++) {
        tu_gpu += transposeGPU(A, C, N, M);
    }

    float kernelBandwidth = 2.0f * 1000.0f * memsz/(1024*1024*1024)/(tu_gpu/NUM_REPS);
    printf("transpose GPU, Throughput = %.4f GB/s, Time = %f ms, Size = %u fp32 elements, NumDevsUsed = %u\n", kernelBandwidth, tu_gpu/NUM_REPS, M*N, 1);

    printf("%s\n", (fcheck(B, C, elnum, th) ? "Test FAILED!" : "Test PASSED"));

    free(C);
    free(B);
    free(A);
}

// main function: process arguments and run tests
int main(int argc, char **argv)
{
    char *names[] = { "m", "n", "reps", "thresh" };
    int flags[] = { 1, 1, 1, 1 };
    int map[] = { 0, 1, 2, 3 };
    int N = 128;
    int M = 128;
    int NUM_REPS = 100;
    struct OptionTable *opttable = make_opttable(4, names, flags, map);
    argproc(argc, argv, opttable);
    printf("%s Starting...\n\n", argv[0]);
    print_gpuinfo(argc, (const char**)argv);

    const char *str_M = opttable->table[0].val;
    const char *str_N = opttable->table[1].val;
    const char *str_REPS = opttable->table[2].val;

    float thresh = 0.0;

    if (str_M) M = atoi(str_M);
    if (str_N) N = atoi(str_N);
    if (str_REPS) NUM_REPS = atoi(str_REPS);

    runtest2d(thresh, N, M, NUM_REPS);
    runtest(thresh, N, M, NUM_REPS);

    free_opttable(opttable);
    return 0;
}
