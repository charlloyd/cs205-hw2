/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuda2acc.h"
#include "device_helper.h"
#include "helper_functions.h"
#include "helper_cuda.h"
#include "helper_math.h"
#include <math.h>

typedef unsigned char uint8;

typedef unsigned short int uint16;

typedef struct{
    unsigned char r, g, b, a;
} RGBA8_misaligned;

typedef struct{
    unsigned int l, a;
} LA32_misaligned;

typedef struct{
    unsigned int r, g, b;
} RGB32_misaligned;


typedef struct{
    unsigned int r, g, b, a;
} RGBA32_misaligned;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
__declspec(align(4)) struct RGBA8{
    unsigned char r, g, b, a;
};
#else
struct RGBA8{
    unsigned char r, g, b, a;
}__attribute__((aligned(4)));
#endif

typedef unsigned int I32;

#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef __declspec(align(8)) struct{
    unsigned int l, a;
} LA32;

typedef __declspec(align(16)) struct {
    unsigned int r, g, b;
} RGB32;

typedef __declspec(align(16)) struct {
    unsigned int r, g, b, a;
} RGBA32;
#else
typedef struct {
    unsigned int l, a;
}__attribute__((aligned(8))) LA32;

typedef struct {
    unsigned int r, g, b;
}__attribute__((aligned(16))) RGB32;

typedef struct {
    unsigned int r, g, b, a;
}__attribute__((aligned(16)))  RGBA32;
#endif


////////////////////////////////////////////////////////////////////////////////
// Because G80 class hardware natively supports global memory operations
// only with data elements of 4, 8 and 16 bytes, if structure size
// exceeds 16 bytes, it can't be efficiently read or written,
// since more than one global memory non-coalescable load/store instructions
// will be generated, even if __align__ option is supplied.
// "Structure of arrays" storage strategy offers best performance 
// in general case. See section 5.1.2 of the Programming Guide.
////////////////////////////////////////////////////////////////////////////////
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
typedef __declspec(align(16)) struct{
    RGBA32 c1, c2;
} RGBA32_2;
#else
typedef struct{
    RGBA32 c1, c2;
} __attribute__((aligned(16))) RGBA32_2;
#endif
#define DEVICE_NUM  2

////////////////////////////////////////////////////////////////////////////////
// Common host and device functions
////////////////////////////////////////////////////////////////////////////////
//Round a / b to nearest higher integer value
int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Round a / b to nearest lower integer value
int iDivDown(int a, int b){
    return a / b;
}

//Align a to nearest higher multiple of b
int iAlignUp(int a, int b){
    return (a % b != 0) ? (a - a % b + b) : a;
}

//Align a to nearest lower multiple of b
int iAlignDown(int a, int b){
    return a - a % b;
}

template<class TData>  void testKernel(TData *d_odata, TData *d_idata, int numElements){
//const int numThreads = 1;
    char *tmp_odata = (char *)d_odata, *tmp_idata = (char *)d_idata;
    #pragma acc kernels loop deviceptr(tmp_idata, tmp_odata) independent
    for(int pos = 0; pos < sizeof(TData) * numElements; pos++){
        tmp_odata[pos] = tmp_idata[pos];
    }
}

template<class TData> int testCPU(TData *h_odata, TData *h_idata, int numElements, int packedElementSize){
    for(int pos = 0; pos < numElements; pos++){
        TData src = h_idata[pos];
        TData dst = h_odata[pos];
        for(int i = 0; i < packedElementSize; i++){
            if (((char *)&src)[i] != ((char *)&dst)[i])  
            return 0;
        }	
    }
    return 1;
}

#ifdef __DEVICE_EMULATION__
const int MEM_SIZE  = 4000000;
const int NUM_ITERATIONS = 1;
#else 
const int MEM_SIZE  = 50000000;
const int NUM_ITERATIONS = 32;
#endif

unsigned char *d_idata, *d_odata;
unsigned char *h_idataCPU, *h_odataGPU;

template< class TData> int runTest(int packedElementSize, int memory_size){
    const int totalMemSizeAligned = iAlignDown(memory_size, sizeof(TData));
    const int numElements = iDivDown(memory_size, sizeof(TData));
    #pragma acc kernels deviceptr(d_odata)
    for (int i = 0; i < memory_size; i++)
    {
        d_odata[i] = 0;
    }

    StartTimer();
    for(int i = 0; i < NUM_ITERATIONS; i++){
        testKernel((TData *)d_odata, (TData *)d_idata, numElements);
    }	
    double gpuTime = GetTimer()/NUM_ITERATIONS;

    printf("Avg. time: %f ms / Copy throughput:%f GB/s.\n", gpuTime,(double)totalMemSizeAligned / (gpuTime * 0.001 * 1073741824.0));

    acc_memcpy_from_device(h_odataGPU, d_odata, memory_size);

    int flag = testCPU((TData *)h_odataGPU, (TData *)h_idataCPU, numElements, packedElementSize);
    printf(flag ? "\tTEST OK\n" : "\tTEST FAILURE\n" );
    return !flag;
}
int main (int argc, char **argv){
    int i, nTotalFailures = 0;	
    printf("[%s] - Starting...\n", argv[0]);

    print_gpuinfo(argc, (const char **)argv);

    acc_init(acc_device_nvidia);

    int devID = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, devID);
    printf("[%s] has %d MP(s) x %d (Cores/MP) = %d (Cores)\n",
           deviceProp.name, deviceProp.multiProcessorCount,
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
           _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);

    float scale_factor = max((192.0f / (_ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * (float)deviceProp.multiProcessorCount)), 1.0f);

    int MemorySize  = (int) (MEM_SIZE / scale_factor) & 0xffffff00;
    printf("> Compute scaling value = %4.2lf\n", scale_factor);
    printf("> Memory Size = %d\n", MemorySize);

    printf("Allocating memory...\n");
    h_idataCPU = (unsigned char *)malloc(MemorySize);
    h_odataGPU = (unsigned char *)malloc(MemorySize);
    d_odata = (unsigned char *)acc_malloc(MemorySize);
    d_idata = (unsigned char *)acc_malloc(MemorySize);

    printf("Generating host input data array...\n");
    for(i = 0; i < MemorySize; i++){
        h_idataCPU[i] = (i & 0xFF) + 1;
    }
    printf("Uploading input data to GPU memory...\n");
    acc_memcpy_to_device(d_idata, h_idataCPU, MemorySize);
    printf("Testing misaligned types...\n");
    printf("uint8...\n");
    nTotalFailures += runTest<uint8>(1, MemorySize);

    printf("uint16...\n");
    nTotalFailures += runTest<uint16>(2, MemorySize);

    printf("RGBA8_misaligned...\n");
    nTotalFailures += runTest<RGBA8_misaligned>(4, MemorySize);

    printf("LA32_misaligned...\n");
    nTotalFailures += runTest<LA32_misaligned>(8, MemorySize);

    printf("RGB32_misaligned...\n");
    nTotalFailures += runTest<RGB32_misaligned>(12, MemorySize);

    printf("RGBA32_misaligned...\n");
    nTotalFailures += runTest<RGBA32_misaligned>(16, MemorySize);

    printf("Testing aligned types...\n");
    printf("RGBA8...\n");
    nTotalFailures += runTest<RGBA8>(4, MemorySize);

    printf("I32...\n");
    nTotalFailures += runTest<I32>(4, MemorySize);

    printf("LA32...\n");
    nTotalFailures += runTest<LA32>(8, MemorySize);

    printf("RGB32...\n");
    nTotalFailures += runTest<RGB32>(12, MemorySize);

    printf("RGBA32...\n");
    nTotalFailures += runTest<RGBA32>(16, MemorySize);

    printf("RGBA32_2...\n");
    nTotalFailures += runTest<RGBA32_2>(32, MemorySize);

    printf("\n[alignedTypes] -> Test Results: %d Failures\n", nTotalFailures);
    printf((nTotalFailures==0) ? "Test passed\n" : "Test failed!\n" );

    printf("Shutting down...\n");
    acc_free(d_idata);
    acc_free(d_odata);
    free(h_odataGPU);
    free(h_idataCPU);
}
