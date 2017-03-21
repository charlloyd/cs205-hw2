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
 * Bandwidth Test:
 *     It tests the bandwidth of data transfering from host to device, device
 *     to device and device to host. It will test a range of data length to
 *     get an average value.
 *     Usage:
 *         "--begin=<N>": Specify the min size of test range, default 1024.
 *         "--end=<N>": Specify the max size of test range, default 102400.
 *         "--inc=<N>": Specify the increment of each test, default 1024.
 *         "--thresh=<th>": Specify the threshold used to check answer is whether passed or failed, default 1.0.
 */

#include "cuda2acc.h"
#include "timer.h"

// there is no device to device memcpy routine in openacc so that
// we use assigning to substitute it
double test_d2d(float *A, float *B, unsigned int N)
{
    StartTimer();
    unsigned int i;
    #pragma acc kernels loop present(A[0:N],B[0:N]) independent
    for (i = 0; i < N; i++) {
        A[i] = B[i];
    }
    return GetTimer();
}

// test and output answer
void runtest(unsigned int begin, unsigned int end, unsigned int inc, float th)
{
    float *A = NULL, *B = NULL;
    float avg_h2d = 0, avg_d2d = 0, avg_d2h = 0;
    int n, cnt = 0, deviceCount;
    unsigned int memsz;
    double tu_h2d, tu_d2d, tu_d2h;
    int iRetVal = 0;

    cudaGetDeviceCount(&deviceCount);

    for (n = begin ; n <= end; n += inc) {
        memsz = sizeof(float) * n;
        A = (float *)malloc(memsz);
        B = (float *)malloc(memsz);
        finit_rand(A, n);

        StartTimer();
        // using directive instead of routine for compatiblity of openacc 1.0
        #pragma acc enter data copyin(A[0:n])
        tu_h2d = GetTimer();
        avg_h2d += calc_bandwidth(memsz, tu_h2d);
        #pragma acc enter data create(B[0:n])
        tu_d2d = test_d2d(B, A, n);
        avg_d2d += calc_bandwidth(memsz, tu_d2d);
        StartTimer();
        #pragma acc exit data copyout(B[0:n])
        tu_d2h = GetTimer();
        avg_d2h += calc_bandwidth(memsz, tu_d2h);
        #pragma acc exit data delete(A[0:n])
        iRetVal += fcheck(A, B, n, th);
        free(A);
        free(B);
        cnt++;
    }

    printf("Host to Device Bandwidth, %d Device(s)\n", deviceCount);
    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    printf("   %u\t\t\t%s%.1f\n\n", memsz, (memsz < 10000)? "\t" : "", avg_h2d / cnt);
    printf("Device to Host Bandwidth, %d Device(s)\n", deviceCount);
    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    printf("   %u\t\t\t%s%.1f\n\n", memsz, (memsz < 10000)? "\t" : "", avg_d2h / (float) cnt);
    printf("Device to Device Bandwidth, %d Device(s)\n", deviceCount);
    printf("   Transfer Size (Bytes)\tBandwidth(MB/s)\n");
    printf("   %u\t\t\t%s%.1f\n\n", memsz, (memsz < 10000)? "\t" : "", avg_d2d / (float) cnt);

    printf("%s\n", (iRetVal==0) ? "Result = PASS" : "Result = FAIL");
}

// main function: process arguments and run tests
int main(int argc, char **argv)
{
    int begin = 1024, end = 1024000, inc = 102400;
    char *names[] = { "begin", "end", "inc", "thresh" };
    int flags[] = { 1, 1, 1, 1 };
    int map[] = { 0, 1, 2, 3 };
    struct OptionTable *opttable = make_opttable(4, names, flags, map);

    printf("[%s] - Starting...\n", argv[0]);

    argproc(argc, argv, opttable);

    if (opttable->table[0].val)
        begin = atoi(opttable->table[0].val);
    if (opttable->table[1].val)
        end = atoi(opttable->table[1].val);
    if (opttable->table[2].val)
        inc = atoi(opttable->table[2].val);
    float th = 1.0;
    if (opttable->table[3].val)
        th = atof(opttable->table[3].val);

    print_gpuinfo(argc, (const char **)argv);

    runtest(begin, end, inc, th);

    free_opttable(opttable);
    return 0;
}
