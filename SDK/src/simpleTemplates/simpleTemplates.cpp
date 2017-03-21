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
 * Simple Templates:
 *     The openacc version of Template. Multiply a vector with a scalar.
 */

#include "cuda2acc.h"
#include "timer.h"

int g_TotalFailures = 0;

// Here's the generic wrapper for cutCompare*
template<class T>
class ArrayComparator
{
    public:
        bool compare(const T *reference, T *data, unsigned int len)
        {
            fprintf(stderr, "Error: no comparison function implemented for this type\n");
            return false;
        }
};

// Here's the specialization for ints:
template<>
class ArrayComparator<int>
{
    public:
        bool compare(const int *reference, int *data, unsigned int len)
        {
            return compareData(reference, data, len, 0.15f, 0.0f);
        }
};

// Here's the specialization for floats:
template<>
class ArrayComparator<float>
{
    public:
        bool compare(const float *reference, float *data, unsigned int len)
        {
            return compareData(reference, data, len, 0.15f, 0.15f);
        }
};

// template verctor multiplication with scalar
template <class T>
void computeCPU(T *B, T *A, unsigned int N)
{
    const T num = static_cast<T>(N);
    StartTimer();
    for (int i = 0; i < N; i++) {
        B[i] = A[i] * num;
    }
}

// openacc template multiplication
template <class T>
void computeGPU(T *C, T *A, unsigned int N)
{
    T num = static_cast<T>(N);

    StartTimer();
    #pragma acc enter data copyin(A[0:N]) create(C[0:N])
    #pragma acc kernels loop present(A[0:N],C[0:N]) independent
    for (int i = 0; i < N; i++) {
        C[i] = A[i] * num;
    }
    printf("Processing time: %f (ms)\n", GetTimer());
    #pragma acc exit data delete(A[0:N]) copyout(C[0:N])
}

// allocate memory and do test
template <class T>
void runtest(unsigned int N)
{
    T *A, *B, *C;
    ArrayComparator<T> comparator;

    A = new T[N];
    B = new T[N];
    C = new T[N];

    // initialization
    for (int i = 0; i < N; i++)
        A[i] = (T) i;
    computeCPU(B, A, N);
    computeGPU(C, A, N);
    bool res = comparator.compare(B, C, N);
    printf("Compare %s\n\n", (1 == res) ? "OK" : "MISMATCH");
    g_TotalFailures += (1 != res);

    delete [] A;
    delete [] B;
    delete [] C;
}

// select which type to execute
void runtest_sel(const char *type, unsigned int n)
{
    if (!type || strcmp(type, "float") == 0) {
        runtest<float>(n);
    } else if (strcmp(type, "int") == 0){
        runtest<int>(n);
    } else {
        runtest<float>(n);
    }
}

// main function: process arguments and call runtest()
int main(int argc, char *argv[])
{
    print_gpuinfo(argc, (const char **)argv);

    printf("> runTest<float,32>\n");
    runtest<float>(32);
    printf("> runTest<int,64>\n");
    runtest<int>(64);

    printf("\n[simpleTemplates] -> Test Results: %d Failures\n", g_TotalFailures);

    return 0;
}
