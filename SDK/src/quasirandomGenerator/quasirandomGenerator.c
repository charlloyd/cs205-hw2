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
#include "timer.h"

typedef long long int INT64;

// macros definition
#define QRNG_DIMENSIONS 3
#define QRNG_RESOLUTION 31
#define INT_SCALE (1.0f / (float)0x80000001U)
#define INT63_SCALE (1.0 / (double)0x8000000000000001ULL)

// tables
extern uint32_t table[QRNG_DIMENSIONS][QRNG_RESOLUTION];
extern int64_t cjn[63][QRNG_RESOLUTION];

// cpu generator
double quasirandomGeneratorCPU(float *result, int n)
{
    uint32_t dim, pos;

    StartTimer();
    // generator codes
    for (dim = 0; dim < QRNG_DIMENSIONS; dim++) {
        for (pos = 0; pos < n; pos++) {
            int64_t tmp = 0, i = pos;
            int bit;
            for(bit = 0; bit < 63; bit++, i >>= 1)
                if(i & 1)
                    tmp ^= cjn[bit][dim];
            result[pos] = (double)(tmp + 1) * INT63_SCALE;
        }
    }
    return GetTimer();
}

double getQuasirandomValue63(INT64 i, int dim)
{
    INT64 result = 0;

    for (int bit = 0; bit < 63; bit++, i >>= 1)
        if (i & 1) result ^= cjn[bit][dim];

    return (double)(result + 1) * INT63_SCALE;
}


// vector addition using openacc
double quasirandomGeneratorGPU(float *result, int n)
{
    uint32_t dim, pos;
    //#pragma acc data copyin(cjn[0:63][0:QRNG_DIMENSIONS]) copyout(result[0:n]) is preferred here. "enter/exit data" is used for timing purpose.
    #pragma acc enter data create(result[0:n]) copyin(cjn[0:63][0:QRNG_DIMENSIONS])
    StartTimer();

    // generator codes
    #pragma acc data present(cjn[0:63][0:QRNG_DIMENSIONS],result[0:n])
    {
    for (dim = 0; dim < QRNG_DIMENSIONS; dim++) {
        #pragma acc kernels loop independent
        for (pos = 0; pos < n; pos++) {
            int64_t tmp = 0, i = pos;
            int bit;
            #pragma acc loop independent
            for(bit = 0; bit < 63; bit++, i >>= 1) {
                if(i & 1)
                    tmp ^= cjn[bit][dim];
            }
            result[pos] = (float)(tmp + 1) * (float)(INT63_SCALE);
        }
    }
    }

    double endtime = GetTimer();
    #pragma acc exit data copyout(result[0:n]) delete(cjn[0:63][0:QRNG_DIMENSIONS])
    return endtime;
}

// run test
void runtest(int N, float thresh)
{
    double gpuTime = 0;
    int numIterations = 20;

    printf("Testing QRNG...\n\n");
    printf("Allocating CPU memory...\n");
    float *cpu_result = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));
    float *gpu_result = (float *)malloc(QRNG_DIMENSIONS * N * sizeof(float));

    double tu_cpu = quasirandomGeneratorCPU(cpu_result, N);
    // warm up
    quasirandomGeneratorGPU(gpu_result, N);

    for (int i = 0; i < numIterations; i++) {
        gpuTime += quasirandomGeneratorGPU(gpu_result, N);
    }

    gpuTime = gpuTime/(double)numIterations*1e-3;

    printf("quasirandomGenerator, Throughput = %.4f GNumbers/s, Time = %.5f s, Size = %u Numbers, NumDevsUsed = %u, Workgroup = %u\n\n",
           (double)QRNG_DIMENSIONS * (double)N * 1.0E-9 / gpuTime, gpuTime, QRNG_DIMENSIONS*N, 1, 128*QRNG_DIMENSIONS);

    printf("Comparing to the CPU results...\n\n");
    printf("%s\n", (fcheck(gpu_result, cpu_result, N, thresh) ? "Test FAILES" : "Test PASSES"));

    free(cpu_result);
    free(gpu_result);
}

// main function: process arguments and call runtest()
int main(int argc, char **argv)
{
    float th = 1.0;
    int n = 1048576;

    char *names[] = { "type", "thresh", "n" };
    int flags[] = { 1, 1, 1 };
    int map[] = { 0, 1, 2 };
    struct OptionTable *opttable = make_opttable(3, names, flags, map);
    argproc(argc, argv, opttable);

    const char *str_type = opttable->table[0].val, *str_th = opttable->table[1].val;

    printf("%s Starting...\n\n", argv[0]);

    if (str_type && strcmp(str_type, "double") == 0)
        ;
    if (str_th)
    th = atof(str_th);
    if (opttable->table[2].val)
        n = atoi(opttable->table[2].val);

    print_gpuinfo(argc, (const char **)argv);

    runtest(n, th);

    free_opttable(opttable);
    return 0;
}

// init tables
uint32_t table[QRNG_DIMENSIONS][QRNG_RESOLUTION] = {
    { 0x40000000, 0x20000000, 0x10000000, 0x8000000, 0x4000000, 0x2000000, 0x1000000, 0x800000, 0x400000, 0x200000, 0x100000, 0x80000, 0x40000, 0x20000, 0x10000, 0x8000, 0x4000, 0x2000, 0x1000, 0x800, 0x400, 0x200, 0x100, 0x80, 0x40, 0x20, 0x10, 0x8, 0x4, 0x2, 0x1 },
    { 0x40000000, 0x60000000, 0x50000000, 0x78000000, 0x44000000, 0x66000000, 0x55000000, 0x7f800000, 0x40400000, 0x60600000, 0x50500000, 0x78780000, 0x44440000, 0x66660000, 0x55550000, 0x7fff8000, 0x40004000, 0x60006000, 0x50005000, 0x78007800, 0x44004400, 0x66006600, 0x55005500, 0x7f807f80, 0x40404040, 0x60606060, 0x50505050, 0x78787878, 0x44444444, 0x66666666, 0x55555555 },
    { 0x60000000, 0x48000000, 0x38000000, 0x7a000000, 0x5e000000, 0x36800000, 0x65800000, 0x4b200000, 0x3e600000, 0x7ec80000, 0x5db80000, 0x315a0000, 0x603e0000, 0x487e8000, 0x385d8000, 0x7a312000, 0x5e606000, 0x36c84800, 0x65b83800, 0x4b5a7a00, 0x3e3e5e00, 0x7efeb680, 0x5ddde580, 0x31116b20, 0x60005e60, 0x480036c8, 0x380065b8, 0x7a004b5a, 0x5e003e3e, 0x36807efe, 0x65805ddd }
};
int64_t cjn[63][QRNG_RESOLUTION] = {
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000 },
    { 0x0, 0x0, 0x0, 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000 },
    { 0x0, 0x0, 0x80000000, 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000 },
    { 0x0, 0x0, 0x80000000, 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000 },
    { 0x80000000, 0x80000000, 0x20000000, 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000 },
    { 0x40000000, 0x40000000, 0x60000000, 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000 },
    { 0x20000000, 0x60000000, 0x48000000, 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000 },
    { 0x10000000, 0x50000000, 0x38000000, 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000 },
    { 0x8000000, 0x78000000, 0x7a000000, 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000 },
    { 0x4000000, 0x44000000, 0x5e000000, 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000 },
    { 0x2000000, 0x66000000, 0x36800000, 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000 },
    { 0x1000000, 0x55000000, 0x65800000, 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000 },
    { 0x800000, 0x7f800000, 0x4b200000, 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000 },
    { 0x400000, 0x40400000, 0x3e600000, 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000 },
    { 0x200000, 0x60600000, 0x7ec80000, 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800 },
    { 0x100000, 0x50500000, 0x5db80000, 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400 },
    { 0x80000, 0x78780000, 0x315a0000, 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200 },
    { 0x40000, 0x44440000, 0x603e0000, 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100 },
    { 0x20000, 0x66660000, 0xc87e8000, 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80 },
    { 0x10000, 0x55550000, 0xb85d8000, 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40 },
    { 0x8000, 0x7fff8000, 0x5a312000, 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20 },
    { 0x4000, 0x40004000, 0x3e606000, 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10 },
    { 0x2000, 0x60006000, 0x7ec84800, 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8 },
    { 0x1000, 0x50005000, 0x5db83800, 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4 },
    { 0x800, 0x78007800, 0x315a7a00, 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2 },
    { 0x400, 0x44004400, 0x603e5e00, 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1 },
    { 0x200, 0x66006600, 0x487eb680, 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0 },
    { 0x100, 0x55005500, 0x385de580, 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0 },
    { 0x80, 0x7f807f80, 0x7a316b20, 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x40, 0x40404040, 0x5e605e60, 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
{ 0x20, 0x60606060, 0x36c836c8, 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x10, 0x50505050, 0x65b865b8, 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x8, 0x78787878, 0x4b5a4b5a, 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x4, 0x44444444, 0x3e3e3e3e, 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x2, 0x66666666, 0x7efefefe, 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 },
    { 0x1, 0x55555555, 0x5ddddddd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 }
};
