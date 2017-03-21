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
 * Scalar Production:
 *     Perform a scalar production on VECTOR_N arrays of ELEMENT_N values to produce an array of answers.
 *     Usage:
 *         "--VECTOR_N=<N>": Specify the number of arrays to do scalar production.
 *         "--ELEMENT_N=<N>": Specify the number of elements in an array.
 *         "--thresh=<th>": Specify the threshold used to check whether answer is passed.
 */

#include "cuda2acc.h"
#include "timer.h"

unsigned int VECTOR_N = 256;
unsigned int ELEMENT_N = 4096;

// CPU version of scalar production
void scalarProdCPU(float *A, float *B, float *C)
{
    for (int i = 0; i < VECTOR_N; i++) {
        double sum = 0;
        unsigned int start = i * ELEMENT_N, end = start + ELEMENT_N;
        for (int j = start; j < end; j++) {
            sum += A[j] * B[j];
        }
        C[i] = (float)sum;
    }
}

// openacc version of scalar production on float array
void scalarProdGPU(float *A, float *B, float *D)
{
    unsigned int DATA_N = VECTOR_N * ELEMENT_N;

    printf("...copying input data to GPU mem.\n");
    #pragma acc enter data copyin(A[0:DATA_N],B[0:DATA_N]) create(D[0:VECTOR_N])
    printf("Executing GPU kerELEMENT_N...\n");
    StartTimer();
    #pragma acc data present(A[0:DATA_N],B[0:DATA_N],D[0:VECTOR_N])
    {
    #pragma acc kernels loop independent
    for (int i = 0; i < VECTOR_N; i++) {
        double sum = 0;
        unsigned int start = i * ELEMENT_N, end = start + ELEMENT_N;
        #pragma acc loop reduction(+:sum)
        for (int j = start; j < end; j++) {
            sum += A[j] * B[j];
        }
        D[i] = (float)sum;
    }
    }

    printf("GPU time used: %f (ms)\n", GetTimer());
    printf("Reading back GPU result...\n");
    #pragma acc exit data delete(A[0:DATA_N],B[0:DATA_N]) copyout(D[0:VECTOR_N])
}

// test and output answer
void runtest(float thresh)
{
    float *A, *B, *C, *D;
    double delta, ref, sum_delta, sum_ref, L1norm;
    unsigned int DATA_N = VECTOR_N * ELEMENT_N;
    int i;

    printf("Initializing data...\n");
    printf("...allocating CPU memory.\n");
    A = (float *)malloc(sizeof(float) * DATA_N);
    B = (float *)malloc(sizeof(float) * DATA_N);
    C = (float *)malloc(sizeof(float) * DATA_N);
    D = (float *)malloc(sizeof(float) * DATA_N);

    printf("...generating input data in CPU mem.\n");
    finit_rand(A, DATA_N);
    finit_rand(B, DATA_N);

    int tu_native, tu_acc;
    printf("...running CPU scalar product calculation\n");
    scalarProdCPU(A, B, C);
    scalarProdGPU(A, B, D);

    printf("...comparing the results\n");
    sum_delta = 0;
    sum_ref = 0;

    for (i = 0; i < VECTOR_N; i++)
    {
        delta = fabs(D[i] - C[i]);
        ref   = C[i];
        sum_delta += delta;
        sum_ref   += ref;
    }

    L1norm = sum_delta / sum_ref;

    printf("Shutting down...\n");
    free(A);
    free(B);
    free(C);
    free(D);

    printf("L1 error: %E\n", L1norm);
    printf((L1norm < 1e-6) ? "Test passed\n" : "Test failed!\n");
    exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
}

// main function: process arguments and run tests
int main(int argc, char *argv[])
{
    float thresh = 1e-6;
   
    char *names[] = { "VECTOR_N", "ELEMENT_N", "thresh" };
    int flags[] = { 1, 1, 1 };
    int map[] = { 0, 1, 2 };
    struct OptionTable *opttable = make_opttable(3, names, flags, map);
    argproc(argc, argv, opttable);
    if (opttable->table[0].val)
        VECTOR_N = atoi(opttable->table[0].val);
    if (opttable->table[1].val)
        ELEMENT_N = atoi(opttable->table[1].val);
    if (opttable->table[2].val)
        thresh = atof(opttable->table[2].val);

    print_gpuinfo(argc, (const char**)argv);
    runtest(thresh);

    free_opttable(opttable);
    return 0;
}
