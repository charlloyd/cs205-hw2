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
 * Matrix multiplication: C = A * B.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H__
#include "cuda2acc.h"
#include "matrixMul.h"
#include "timer.h"

#define CHECK_BANK_CONFLICTS 0
#define AS(i, j) cutilBankChecker(((float*)&As[0][0]), (BLOCK_SIZE * i + j))                                                                    
#define BS(i, j) cutilBankChecker(((float*)&Bs[0][0]), (BLOCK_SIZE * i + j))
#else
#define AS(i, j) As[i][j]
#define BS(i, j) Bs[i][j]
#endif
static char *sSDKsample = "matrixMul";

void matrixMulCPU(float* C, const float* A, const float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    for (unsigned int i = 0; i < hA; ++i) {
        for (unsigned int j = 0; j < wB; ++j) {
            float sum = 0;
            for (unsigned int k = 0; k < wA; ++k) {
                float a = A[i * wA + k];
                float b = B[k * wB + j];
                sum += a * b;
            }
            C[i * wB + j] = sum;
        }
    }
}

double matrixMulGPU(float* C, float *A, float* B, unsigned int hA, unsigned int wA, unsigned int wB)
{
    #pragma acc enter data copyin(A[0:hA * wA],B[0:wA * wB])
    #pragma acc enter data create(C[0:hA * wB])
    StartTimer();
    #pragma acc data present(A[0:hA * wA], B[0:wA * wB], C[0:hA * wB])
    {
        #pragma acc kernels loop independent
        for (unsigned int i = 0; i < hA; ++i) {
            #pragma acc loop independent
            for (unsigned int j = 0; j < wB; ++j) {
                float sum = 0;

                #pragma acc loop independent reduction(+:sum)
                for (unsigned int k = 0; k < wA; ++k) {
                    float a = A[i * wA + k];
                    float b = B[k * wB + j];
                    sum += a * b;
                }
                C[i * wB + j] = sum;
            }
        }
    }
    double endtime = GetTimer();
    #pragma acc exit data copyout(C[0:hA * wB])
    #pragma acc exit data delete(A[0:hA * wA],B[0:wA * wB])
    return endtime;
}

// Allocates a matrix with random float entries.
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
        data[i] = (float) ((int) (11.0f * rand() / (float)RAND_MAX - 5.5f));
}




void printDiff(float *data1, float *data2, int width, int height, int iListLength, float fListTol)
{
    printf("Listing first %d Differences > %.6f...\n", iListLength, fListTol);
    int i,j,k;
    int error_count=0;
    for (j = 0; j < height; j++) 
    {
        if (error_count < iListLength) {
            printf("\n  Row %d:\n", j);
        }
        for (i = 0; i < width; i++) {
            k = j * width + i;
            float fDiff = fabs(data1[k] - data2[k]);
            if (fDiff > fListTol) {                
                if (error_count < iListLength) {
                    printf("    Loc(%d,%d)\tCPU=%.5f\tGPU=%.5f\tDiff=%.6f\n", i, j, data1[k], data2[k], fDiff);
                }
                error_count++;
            }
        }
    }
    printf(" \n  Total Errors = %d\n\n", error_count);
}


void runTest(int argc , char ** argv)
{
    srand(2006);
    unsigned int uiWA, uiHA, uiWB, uiHB, uiWC, uiHC;
    int iSizeMultiple = 1;

    //iSizeMultiple = CLAMP(iSizeMultiple, 1, 10);
    uiWA = WA * iSizeMultiple;
    uiHA = HA * iSizeMultiple;
    uiWB = WB * iSizeMultiple;
    uiHB = HB * iSizeMultiple;
    uiWC = WC * iSizeMultiple;
    uiHC = HC * iSizeMultiple;
    printf("\nUsing Matrix Sizes: A(%u x %u), B(%u x %u), C(%u x %u)\n\n", uiHA, uiWA, uiHB, uiWB, uiHC, uiWC);

    // allocate host memory for matrices A and B
    int size_A = uiWA * uiHA;
    unsigned int mem_size_A = sizeof(float) * size_A;
    float* h_A = (float*)malloc(mem_size_A);
    unsigned int size_B = uiWB * uiHB;
    unsigned int mem_size_B = sizeof(float) * size_B;
    float* h_B = (float*)malloc(mem_size_B);

    // initialize host memory
    randomInit(h_A, size_A);
    randomInit(h_B, size_B);
    unsigned int size_C = uiWC * uiHC;
    unsigned int mem_size_C = sizeof(float) * size_C;
    float* h_C;
    h_C = (float *)malloc(mem_size_C);

    acc_init(acc_device_nvidia);

    // allocate host memory for the result
    int nIter = 300;
    double totaltime = 0;
    printf("Computing result using OpenACC\n");
    for (int j = 0; j < nIter; j++) {
        totaltime += matrixMulGPU(h_C, h_A, h_B, uiHA, uiWA, uiWB);
    }
    double msecPerMatrixMul = totaltime / (double)nIter;
    double flopsPerMatrixMul = 2.0 * (double) uiWA * (double)uiHA * (double) uiWB;
    double gigaFlops = (flopsPerMatrixMul * 1.0e-9f) / (msecPerMatrixMul / 1000.0f);
    printf("matrxiMul, Performance = %.2f GFlops/s, Time = %.3f msec, Size = %.0f Ops\n", gigaFlops, msecPerMatrixMul, flopsPerMatrixMul);

    // compute reference solution
    printf("\nChecking computed result for correctness: Result = ");    
    float* reference = (float*)malloc(mem_size_C);
    matrixMulCPU(reference, h_A, h_B, uiHA, uiWA, uiWB);

    // check result
    printf("%s\n", (fcheck(reference, h_C,size_C, 1.0e-4) ? "FAILED" : "PASSED"));   
    if (fcheck(reference, h_C,size_C, 1.0e-4)) {
        printDiff(reference, h_C, uiWC, uiHC, 100, 1.0e-4f);
    }

    // clean up memory
    free(h_A);
    free(h_B);
    free(h_C);
    free(reference);
}

int main(int argc, char **argv )
{
    printf("[ %s ]\n", sSDKsample);
    printf("%s Starting...\n\n", argv[0]);

    print_gpuinfo(argc, (const char **)argv);
    runTest(argc, argv);
    return 0;
}
