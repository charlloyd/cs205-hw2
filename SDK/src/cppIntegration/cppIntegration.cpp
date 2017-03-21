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

unsigned int num_threads = 0;

void computeGold(char *reference, char *idata, const unsigned int len)
{
    for (unsigned int i = 0; i < len; i++) {
        reference[i] = idata[i] - 10;
    }
}

void computeGold2(int2* reference, int2* idata, const unsigned int len)
{
    for (unsigned int i = 0; i < len; i++) {
        reference[i].x = idata[i].x - idata[i].y;
        reference[i].y = idata[i].y;
    }
}

void computeGold_GPU(int *g_data)
{
    #pragma acc kernels loop deviceptr(g_data) independent
    for (int i = 0; i < 16; i++) {
        int data = g_data[i];
        g_data[i] = ((((data <<  0) >> 24) - 10) << 24)
                    | ((((data <<  8) >> 24) - 10) << 16)
                    | ((((data << 16) >> 24) - 10) <<  8)
                    | ((((data << 24) >> 24) - 10) <<  0);
    }
}

void computeGold2_GPU(int2 * g_data)
{
    #pragma acc kernels loop deviceptr(g_data) independent
    for(int i = 0; i < 16; i++) {
        int2 data = g_data[i];
        g_data[i].x = data.x - data.y;
    }
}

void runTest(const int argc, const char ** argv, char * data, int2 * data_int2, unsigned int len)
{
    num_threads = len / 4;

    const unsigned int mem_size = sizeof(char) * len;
    const unsigned int mem_size_int2 = sizeof(int2) * len;

    char * reference = (char*) malloc(mem_size);
    computeGold(reference, data, len);
    int2 * reference2 = (int2*)malloc(mem_size_int2);
    computeGold2(reference2, data_int2, len);

    #pragma acc data copy(data[0:len], data_int2[0:len])
    {
        #pragma acc host_data use_device(data)
        computeGold_GPU((int *)data);
        #pragma acc host_data use_device(data_int2)
        computeGold2_GPU(data_int2);
    }

    bool success = true;
    for (unsigned int i = 0; i < len; i++) {
        if (reference[i] != data[i] || reference2[i].x != data_int2[i].x || reference2[i].y != data_int2[i].y) {
            success = false;
        }
    }
    printf("%s\n", success ? "Test PASSES" : "Test FAILS");

    free(reference);
    free(reference2);
}

int main (int argc, char ** argv)
{
    int len = 16;
    char str[] = {82, 111, 118,118,121,42, 97,121,124,118,110,56,10,10,10,10};
    int2 i2[16];

    print_gpuinfo(argc, (const char **)argv);

    for(int i = 0; i < len; i++) {
        i2[i].x = str[i];
        i2[i].y = 10;
    }

    runTest(argc, (const char**)argv, str, i2,len);

    printf("%s\n", str);
    for (int i = 0; i < len; i++) {
        printf("%c", (char)i2[i].x);
    }
    printf("\n");
}
