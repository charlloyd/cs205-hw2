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

// query number of different type devices
void deviceQueryAcc()
{
    int num_nvidia = acc_get_num_devices(acc_device_nvidia);
    int num_radeon = acc_get_num_devices(acc_device_radeon);
    int num_xeonphi = acc_get_num_devices(acc_device_xeonphi);
    printf("nvidia devices: %d\n", num_nvidia);
    printf("radeon devices: %d\n", num_radeon);
    printf("xeonphi devices: %d\n", num_xeonphi);
    printf("\n");

    // for cuda platforms
    acc_set_device_type(acc_device_nvidia);
    float arr[1024];
    finit_rand(arr, 1024);
    #pragma acc kernels loop independent copy(arr[0:1024])
    for (int i = 0; i < 1024; i++) {
        arr[i] = i;
    }

    printf("cuda device is: %p\n", acc_get_current_cuda_device());
    printf("cuda context is: %p\n", acc_get_current_cuda_context());
    printf("cuda stream is: %p\n", acc_get_cuda_stream(0));
    printf("\n");

    printf("opencl device is: %p\n", acc_get_current_opencl_device());
    printf("opencl context is: %p\n", acc_get_current_opencl_context());
    printf("opencl queue is: %p\n", acc_get_opencl_queue(0));
}

// main function: process arguments and call runtest()
int main(int argc, char **argv)
{
    print_gpuinfo(argc, (const char **)argv);
    deviceQueryAcc();
    return 0;
}
