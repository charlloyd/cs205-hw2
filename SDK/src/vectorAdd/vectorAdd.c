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
 * Vector addition: C = A + B
 *
 * This sample is a very basic sample that implements element by element
 * vector addition.
 *
 * Perform vector addition on a float array using openacc.
 * Usage:
 *     "--n=<N>": Specify the number of elements to reduce (default 50000)
 *     "--thresh=<N>": The threshold used to check answer, default 1e-5.
 */

#include "cuda2acc.h"

// CPU vector addition
void vectorAddCPU(float *A, float *B, float * C, uint32_t N)
{
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
}

// vector addition using openacc
void vectorAddGPU(float *A, float *B, float * C, uint32_t N)
{
    #pragma acc enter data copyin(A[0:N],B[0:N])
    printf("Copy input data from the host memory to the CUDA device\n");
    #pragma acc enter data create(C[0:N])
    printf("CUDA kernel launch\n");
    #pragma acc kernels loop present(A[0:N],B[0:N],C[0:N]) independent
    for (int i = 0; i < N; i++) {
        C[i] = A[i] + B[i];
    }
    printf("Copy output data from the CUDA device to the host memory\n");
    #pragma acc exit data copyout(C[0:N])
    #pragma acc exit data delete(A[0:N],B[0:N])
}

// run test
void runtest(uint32_t N, float th)
{
    float *A, *B, *C, *D;

    // Allocate the host vectors
    A = (float *)malloc(sizeof(float) * N);
    B = (float *)malloc(sizeof(float) * N);
    C = (float *)malloc(sizeof(float) * N);
    D = (float *)malloc(sizeof(float) * N);
    finit_rand(A, N);
    finit_rand(B, N);
    vectorAddCPU(A, B, C, N);
    vectorAddGPU(A, B, D, N);

    printf("%s\n", (fcheck(C, D, N, th) ? "Test failed!" : "Test passed"));

    // Free host memory
    free(D);
    free(A);
    free(B);
    free(C);
}

// main function: process arguments and call runtest()
int main(int argc, char **argv)
{
    unsigned int n = 50000;
    float th = 1e-5;

    char *names[] = { "n", "thresh" };
    int flags[] = { 1, 1 };
    int map[] = { 0, 1 };
    struct OptionTable *opttable = make_opttable(2, names, flags, map);
    argproc(argc, argv, opttable);

    const char *str_n = opttable->table[0].val, *str_th = opttable->table[1].val;
    if (str_n)
        n = atoi(str_n);
    if (str_th)
        th = atof(str_th);

    // Print the vector length to be used.
    printf("[Vector addition of %d elements]\n", n);
    print_gpuinfo(argc, (const char **)argv);
    runtest(n, th);

    free_opttable(opttable);
    printf("Done\n");
    return 0;
}
