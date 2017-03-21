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
 * FDTD3d
 * -------
 * This sample applies a finite differences time domain progression stencil 
 * on a 3D surface.
 */

#include "cuda2acc.h"

#include <iostream>
#include <iomanip>

#include <helper_cuda.h>
#include <helper_functions.h>

#include "FDTD3d.h"

#ifndef CLAMP
#define CLAMP(a, min, max) ( MIN(max, MAX(a, min)) )
#endif

// Forward declarations
void showHelp(const int argc, const char **argv);

// FDTD3d cpu version
bool fdtdReference(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{

    const int     outerDimx    = dimx + 2 * radius;
    const int     outerDimy    = dimy + 2 * radius;
    const int     outerDimz    = dimz + 2 * radius;
    const size_t  volumeSize   = outerDimx * outerDimy * outerDimz;
    const int     stride_y     = outerDimx;
    const int     stride_z     = stride_y * outerDimy;
    float        *intermediate = 0;
    const float  *bufsrc       = 0;
    float        *bufdst       = 0;
    float        *bufdstnext   = 0;

    // Allocate temporary buffer
    printf(" calloc intermediate\n");
    intermediate = (float *)malloc(volumeSize * sizeof(float));

    // Decide which buffer to use first (result should end up in output)
    if ((timesteps % 2) == 0) {
        bufsrc     = input;
        bufdst     = intermediate;
        bufdstnext = output;
    } else {
        bufsrc     = input;
        bufdst     = output;
        bufdstnext = intermediate;
    }

    // Run the FDTD (naive method)
    printf(" Host FDTD loop\n");

    for (int it = 0 ; it < timesteps ; it++) {
        printf("\tt = %d\n", it);
        const float *src = bufsrc;
        float *dst       = bufdst;

        for (int iz = -radius ; iz < dimz + radius ; iz++) {
            for (int iy = -radius ; iy < dimy + radius ; iy++) {
                for (int ix = -radius ; ix < dimx + radius ; ix++) {
                    if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz) {
                        float value = (*src) * coeff[0];
                        for(int ir = 1 ; ir <= radius ; ir++) {
                            value += coeff[ir] * (*(src + ir) + *(src - ir));                       // horizontal
                            value += coeff[ir] * (*(src + ir * stride_y) + *(src - ir * stride_y)); // vertical
                            value += coeff[ir] * (*(src + ir * stride_z) + *(src - ir * stride_z)); // in front & behind
                        }
                        *dst = value;
                    } else {
                        *dst = *src;
                    }
                    ++dst;
                    ++src;
                }
            }
        }
        // Rotate buffers
        float *tmp = bufdst;
        bufdst     = bufdstnext;
        bufdstnext = tmp;
        bufsrc = (const float *)tmp;
    }

    printf("\n");

    if (intermediate)
        free(intermediate);

    return true;
}

// FDTD3d openacc version
bool fdtdGPU(float *output, const float *input, const float *coeff, const int dimx, const int dimy, const int dimz, const int radius, const int timesteps)
{
    const int     outerDimx    = dimx + 2 * radius;
    const int     outerDimy    = dimy + 2 * radius;
    const int     outerDimz    = dimz + 2 * radius;
    const size_t  volumeSize   = outerDimx * outerDimy * outerDimz;
    const int     stride_y     = outerDimx;
    const int     stride_z     = stride_y * outerDimy;
    const float  *bufsrc       = NULL;
    float        *bufdst       = NULL;
    float        *bufdstnext   = NULL;

    float *d_input = (float *)acc_malloc(sizeof(float) * volumeSize);
    acc_memcpy_to_device(d_input, (void *)input, volumeSize * sizeof(float));
    float *d_output = (float *)acc_malloc(sizeof(float) * volumeSize);
    float *d_intermediate = (float *)acc_malloc(sizeof(float) * volumeSize);
    acc_copyin((void *)coeff, sizeof(float) * (radius + 1));

    // Decide which buffer to use first (result should end up in output)
    if ((timesteps % 2) == 0) {
        bufsrc     = d_input;
        bufdst     = d_intermediate;
        bufdstnext = d_output;
    } else {
        bufsrc     = d_input;
        bufdst     = d_output;
        bufdstnext = d_intermediate;
    }

    // Execute the FDTD
    printf(" GPU FDTD loop\n");

    for (int it = 0 ; it < timesteps ; it++) {
        printf("\tt = %d ", it);

        // Launch the kernel
        printf("launch kernel\n");

        const float *src = bufsrc;
        float *dst       = bufdst;
        #pragma acc kernels loop independent deviceptr(src,dst) present(coeff[0:radius+1])
        for (int iz = -radius ; iz < dimz + radius ; iz++) {
            #pragma acc loop independent
            for (int iy = -radius ; iy < dimy + radius; iy++) {
                #pragma acc loop independent
                for (int ix = -radius ; ix < dimx + radius ; ix++) {
                    const int ig = (iz + radius) * stride_z + (iy + radius) * stride_y + (ix + radius);
                    if (ix >= 0 && ix < dimx && iy >= 0 && iy < dimy && iz >= 0 && iz < dimz) {
                        float value = 0;
                        #pragma acc loop independent reduction(+:value)
                        for(int ir = 1 ; ir <= radius; ir++) {
                            value += coeff[ir] * (src[ig + ir] + src[ig - ir]);// horizontal
                            value += coeff[ir] * (src[ig + ir * stride_y] + src[ig - ir * stride_y]); // vertical
                            value += coeff[ir] * (src[ig + ir * stride_z] + src[ig - ir * stride_z]); // in front & behind
                        }
                        dst[ig] = value + src[ig] * coeff[0];
                    } else {
                        dst[ig] = src[ig];
                    }
                }
            }
        }

        // Rotate buffers
        float *tmp = bufdst;
        bufdst     = bufdstnext;
        bufdstnext = tmp;
        bufsrc = (const float *)tmp;
    }

    printf("\n");

    acc_delete((void *)coeff, sizeof(float) * (radius + 1));
    acc_memcpy_from_device(output, d_output, sizeof(float) * volumeSize);
    acc_free(d_intermediate);
    acc_free(d_input);
    acc_free(d_output);

    return true;
}

void showHelp(const int argc, const char **argv)
{
    if (argc > 0)
        std::cout << std::endl << argv[0] << std::endl;

    std::cout << std::endl << "Syntax:" << std::endl;
    std::cout << std::left;
    std::cout << "    " << std::setw(20) << "--device=<device>" << "Specify device to use for execution" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimx=<N>" << "Specify number of elements in x direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimy=<N>" << "Specify number of elements in y direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--dimz=<N>" << "Specify number of elements in z direction (excluding halo)" << std::endl;
    std::cout << "    " << std::setw(20) << "--radius=<N>" << "Specify radius of stencil" << std::endl;
    std::cout << "    " << std::setw(20) << "--timesteps=<N>" << "Specify number of timesteps" << std::endl;
    std::cout << "    " << std::setw(20) << "--block-size=<N>" << "Specify number of threads per block" << std::endl;
    std::cout << std::endl;
    std::cout << "    " << std::setw(20) << "--noprompt" << "Skip prompt before exit" << std::endl;
    std::cout << std::endl;
}

// run test
void runtest(int argc, const char **argv)
{
    int defaultDim;
    int dimx;
    int dimy;
    int dimz;
    int outerDimx;
    int outerDimy;
    int outerDimz;
    int radius;
    int timesteps;
    size_t volumeSize;
    memsize_t memsize;

    print_gpuinfo(argc, (const char **)argv);

    // Determine default dimensions
    printf("Set-up, based upon target device GMEM size...\n");

    // Get the memory size of the target device
    printf(" getTargetDeviceGlobalMemSize\n");
    getTargetDeviceGlobalMemSize(&memsize, argc, argv);

    // We can never use all the memory so to keep things simple we aim to
    // use around half the total memory
    memsize /= 2;

    // Most of our memory use is taken up by the input and output buffers -
    // two buffers of equal size - and for simplicity the volume is a cube:
    //   dim = floor( (N/2)^(1/3) )
    defaultDim = (int)floor(pow((memsize / (2.0 * sizeof(float))), 1.0/3.0));

    // By default, make the volume edge size an integer multiple of 128B to
    // improve performance by coalescing memory accesses, in a real
    // application it would make sense to pad the lines accordingly
    int roundTarget = 128 / sizeof(float);
    defaultDim = defaultDim / roundTarget * roundTarget;
    defaultDim -= k_radius_default * 2;

    // Check dimension is valid
    if (defaultDim < k_dim_min)
    {
        printf("insufficient device memory (maximum volume on device is %d, must be between %d and %d).\n", defaultDim, k_dim_min, k_dim_max);
        exit(EXIT_FAILURE);
    }
    else if (defaultDim > k_dim_max)
    {
        defaultDim = k_dim_max;
    }

    // Set default dim
    dimx = defaultDim;
    dimy = defaultDim;
    dimz = defaultDim;
    radius    = k_radius_default;
    timesteps = k_timesteps_default;

    // Parse comand line arguments
    if (checkCmdLineFlag(argc, argv, "dimx"))
    {
        dimx = CLAMP(getCmdLineArgumentInt(argc, argv, "dimx"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimy"))
    {
        dimy = CLAMP(getCmdLineArgumentInt(argc, argv, "dimy"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "dimz"))
    {
        dimz = CLAMP(getCmdLineArgumentInt(argc, argv, "dimz"), k_dim_min, k_dim_max);
    }

    if (checkCmdLineFlag(argc, argv, "radius"))
    {
        radius = CLAMP(getCmdLineArgumentInt(argc, argv, "radius"), k_radius_min, k_radius_max);
    }

    if (checkCmdLineFlag(argc, argv, "timesteps"))
    {
        timesteps = CLAMP(getCmdLineArgumentInt(argc, argv, "timesteps"), k_timesteps_min, k_timesteps_max);
    }

    // Determine volume size
    outerDimx = dimx + 2 * radius;
    outerDimy = dimy + 2 * radius;
    outerDimz = dimz + 2 * radius;
    volumeSize = outerDimx * outerDimy * outerDimz;

    float *input = (float *)malloc(sizeof(float) * volumeSize);
    float *output = (float *)malloc(sizeof(float) * volumeSize);
    float *acc_output = (float *)malloc(sizeof(float) * volumeSize);
    float *coeff = (float *)malloc(sizeof(float) * (radius + 1));

    finit_rand(input, volumeSize);
    for (int i = 0 ; i <= radius; i++)
        coeff[i] = 0.1f;

    // Generate data
    printf("\n\nFDTD on %d x %d x %d volume with symmetric filter radius %d for %d timesteps...\n\n", dimx, dimy, dimz, radius, timesteps);

    // Execute on the host
    printf("fdtdReference...\n");
    fdtdReference(output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    printf("fdtdReference complete\n");

    // Execute on the device
    printf("fdtdGPU...\n");
    fdtdGPU(acc_output, input, coeff, dimx, dimy, dimz, radius, timesteps);
    printf("fdtdGPU complete\n");

    float tolerance = 0.0001f;
    printf("\nCompareData (tolerance %f)...\n", tolerance);
    printf("%s\n", fcheck(output, acc_output, volumeSize, tolerance) ? "FAILED" : "PASSED");

    free(coeff);
    free(output);
    free(acc_output);
    free(input);
}

// main function: process arguments and call runtest()
int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    // Check help flag
    if (checkCmdLineFlag(argc, (const char **)argv, "help"))
    {
        printf("Displaying help on console\n");
        showHelp(argc, (const char **)argv);
    }
    else
    {
        // Execute
        runtest(argc, (const char **)argv);
    }

    return 0;
}
