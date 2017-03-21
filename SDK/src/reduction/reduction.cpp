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
 * Sum Reduction:
 *     Perform a reduction operation on an array of values to produce a single value.
 *     Usage:
 *         "--shmoo":      Test performance for 1 to 32M elements with each of the 7 different kernels
 *         "--n=<N>":      Specify the number of elements to reduce (default 1048576)
 *         "--thresh=<N>": The threshold used to check answer, default 1.0.
 *         "-type=<T>:     The datatype for the reduction, where T is "int", "float", or "double" (default int)
 */

#include "cuda2acc.h"
#include "timer.h"

enum ReduceType
{
    REDUCE_INT,
    REDUCE_FLOAT,
    REDUCE_DOUBLE
};

// CPU version of sum reduction
template<class T>
T reductionCPU(T *A, int N)
{
    int i;
    StartTimer();
    T sum = A[0];
    T c = 0;
    for (i = 1; i < N; i++) {
        T y = A[i] - c;
        T t = sum + y;
        c = (t - sum) - y;
        sum = t;
    }
    printf("reductionCPU time used: %f\n", GetTimer());
    return sum;
}

// openacc version of sum reduction on float array
template<class T>
T reductionGPU(T *A, int N)
{
    int i;
    T sum = 0;
    StartTimer();
    //#pragma acc copyin(A[0:N]) is perferred here. "enter/exit data" is used for timing purpose.
    #pragma acc enter data copyin(A[0:N])
    StartTimer();
    #pragma acc kernels loop present(A[0:N]) reduction(+:sum)
    for (i = 0; i < N; i++) {
        sum += A[i];
    }
    double endtime = GetTimer();
    #pragma acc exit data delete(A[0:N])
    printf("reductionGPU time used: %f\n", GetTimer());
    return sum;
}

// rest performance for 1 to 32M elements with each of the 7 different kernels
template <class T>
void shmoo(T th)
{
    T *A;
    int n = 1048576;
    int max_n = 1 << 25;
    for (n = 1; n <= max_n; n <<= 1) {
        A = (T*)malloc(sizeof(T) * n);
        reductionCPU(A, n);
        reductionGPU(A, n);
        free(A);
    }
}

// test and output answer
template <class T>
bool runtest(int N, T th, bool runShmoo, ReduceType datatype)
{
    T *A;
    T cpu_result, gpu_result;
    double diff = 0;

    if (runShmoo) {
        shmoo(th);
    } else {
        A = (T*)malloc(sizeof(T) * N);
        for (int j = 0; j < N; j++) {
            if (datatype == REDUCE_INT) {
                A[j] = (T)(rand() & 0xFF);
            } else {
                A[j] = (rand() & 0xFF) / (T)RAND_MAX;
            }
        }
        cpu_result = reductionCPU(A, N);
        gpu_result = reductionGPU(A, N);

        if (datatype == REDUCE_INT) {
            printf("\nGPU result = %d\n", (int)gpu_result);
            printf("CPU result = %d\n\n", (int)cpu_result);
        } else {
            printf("\nGPU result = %f\n", (double)gpu_result);
            printf("CPU result = %f\n\n", (double)cpu_result);

            if (datatype == REDUCE_FLOAT)
            {
                th = 1e-8 * N;
            }

            diff = fabs((double)gpu_result - (double)cpu_result);
        }

        free(A);

        if (datatype == REDUCE_INT) {
            return (gpu_result == cpu_result);
        } else {
            return (diff < th);
        }
    }
    return true;
}

// main function: process arguments and run tests
int main(int argc, char *argv[])
{ 
    const char *names[] = { "shmoo", "n", "thresh", "type" };
    int flags[] = { 0, 1, 1, 1 };
    int map[] = { 0, 1, 2, 3 };
    struct OptionTable *opttable = make_opttable(4, names, flags, map);
    bool runShmoo = false;
    bool bResult = false;

    ReduceType datatype = REDUCE_INT;

    print_gpuinfo(argc, (const char **)argv);

    argproc(argc, argv, opttable);

    if (opttable->table[0].val) {
        runShmoo = true;
    }
    const char *str_n = opttable->table[1].val, *str_th = opttable->table[2].val, *str_type = opttable->table[3].val;
    int n = 1048576;
    float th = 1e-12;
    if (str_n)
        n = atoi(str_n);
    if (str_th)
        th = atof(str_th);
    if (str_type) {
        if (strcmp(str_type, "float")) {
            datatype = REDUCE_FLOAT;
        } else if (strcmp(str_type, "double")) {
            datatype = REDUCE_DOUBLE;
        } else {
            datatype = REDUCE_INT;
        }
    }

    switch (datatype) {
        default:
        case REDUCE_INT:
            bResult = runtest<int>(n, th, runShmoo, datatype);
            break;
        case REDUCE_FLOAT:
            bResult = runtest<float>(n, th, runShmoo, datatype);
            break;
        case REDUCE_DOUBLE:
            bResult = runtest<double>(n, th, runShmoo, datatype);
            break;
    }

    free_opttable(opttable);
    printf(bResult ? "Test PASSED\n" : "Test FAILED!\n");
    return 0;
}
