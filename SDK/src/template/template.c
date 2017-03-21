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
 * template:
 *     The prototype of Simple Template.
 *     Usage:
 *         "--n=<N>": Specify the number of elements, default 1048576.
 *         "--thresh=<N>": The threshold used to check answer, default 1.0.
 */

#include "cuda2acc.h"
#include "timer.h"

// multiply a vector with a scalar
void computeCPU(float *A, float *B, unsigned int N)
{
    const float num = (float)(N);
    for (int i = 0; i < N; i++) {
        A[i] = B[i] * num;
    }
}

// multiply a vector with a scalar using openacc
void computeGPU(float *A, float *B, unsigned int N)
{
    float num = (float)(N);
    unsigned int i;
    StartTimer();
    #pragma acc enter data create(A[0:N]) copyin(B[0:N])
    #pragma acc data present(A[0:N],B[0:N])
    {
    #pragma acc kernels loop independent
    for (i = 0; i < N; i++) {
        A[i] = B[i] * num;
    }
    }
    double gettimer = GetTimer();
    #pragma acc exit data copyout(A[0:N]) delete(B[0:N])
    printf("GPU processing time: %f (ms)\n", GetTimer());
}

// test and output answer
void runtest(unsigned int N, float th)
{
    float *A, *B, *C;
    unsigned int size = sizeof(float) * N;
    A = (float*)malloc(size);
    B = (float*)malloc(size);
    // initialization
    finit_rand(A, N);
    computeCPU(B, A, N);

    C = (float*)malloc(size);
    computeGPU(C, A, N);
    printf("%s\n", (fcheck(C, B, N, th) ? "Test FAILED!" : "Test PASSED"));
    free(C);
    free(B);
    free(A);
}


// main function: process arguments and run tests
int main(int argc, char *argv[])
{
    char *names[] = { "n", "thresh" };
    int flags[] = { 1, 1 };
    int map[] = { 0, 1 };
    struct OptionTable *opttable = make_opttable(2, names, flags, map);
    argproc(argc, argv, opttable);

    const char *str_n = opttable->table[0].val, *str_th = opttable->table[1].val;
    unsigned int n = 32;
    float th = 0.0;

    printf("%s Starting...\n\n", argv[0]);

    print_gpuinfo(argc, (const char **)argv);

    if (str_n)
        n = atoi(str_n);
    if (str_th)
        th = atof(str_th);
    runtest(n, th);

    free_opttable(opttable);
    return 0;
}
