/*
 *     Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#ifndef HELPER_CUDA_GL_H
#define HELPER_CUDA_GL_H

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

// includes, graphics
#if defined (__APPLE__) || defined(MACOSX)
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#ifdef __DRIVER_TYPES_H__
#ifndef DEVICE_RESET
#define DEVICE_RESET cudaDeviceReset()
#endif
#else
#ifndef DEVICE_RESET
#define DEVICE_RESET
#endif
#endif

#ifdef __CUDA_GL_INTEROP_H__
////////////////////////////////////////////////////////////////////////////////
// These are CUDA OpenGL Helper functions

inline int gpuGLDeviceInit(int ARGC, const char **ARGV)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    int dev = 0;
    dev = getCmdLineArgumentInt(ARGC, ARGV, "device=");

    if (dev < 0)
    {
        dev = 0;
    }

    if (dev > deviceCount-1)
    {
        fprintf(stderr, "\n");
        fprintf(stderr, ">> %d CUDA capable GPU device(s) detected. <<\n", deviceCount);
        fprintf(stderr, ">> gpuGLDeviceInit (-device=%d) is not a valid GPU device. <<\n", dev);
        fprintf(stderr, "\n");
        return -dev;
    }

    cudaDeviceProp deviceProp;
    checkCudaErrors(cudaGetDeviceProperties(&deviceProp, dev));

    if (deviceProp.computeMode == cudaComputeModeProhibited)
    {
        fprintf(stderr, "Error: device is running in <Compute Mode Prohibited>, no threads can use ::cudaSetDevice().\n");
        return -1;
    }

    if (deviceProp.major < 1)
    {
        fprintf(stderr, "Error: device does not support CUDA.\n");
        exit(EXIT_FAILURE);
    }

    if (checkCmdLineFlag(ARGC, ARGV, "quiet") == false)
    {
        fprintf(stderr, "Using device %d: %s\n", dev, deviceProp.name);
    }

    checkCudaErrors(cudaGLSetGLDevice(dev));
    return dev;
}

// This function will pick the best CUDA device available with OpenGL interop
inline int findCudaGLDevice(int argc, const char **argv)
{
    int devID = 0;

    // If the command-line has a device number specified, use it
    if (checkCmdLineFlag(argc, (const char **)argv, "device"))
    {
        devID = gpuGLDeviceInit(argc, (const char **)argv);

        if (devID < 0)
        {
            printf("no CUDA capable devices found, exiting...\n");
            DEVICE_RESET
            exit(EXIT_SUCCESS);
        }
    }
    else
    {
        // Otherwise pick the device with highest Gflops/s
        devID = gpuGetMaxGflopsDeviceId();
        cudaGLSetGLDevice(devID);
    }

    return devID;
}

////////////////////////////////////////////////////////////////////////////
//! Check for OpenGL error
//! @return bool if no GL error has been encountered, otherwise 0
//! @param file  __FILE__ macro
//! @param line  __LINE__ macro
//! @note The GL error is listed on stderr
//! @note This function should be used via the CHECK_ERROR_GL() macro
////////////////////////////////////////////////////////////////////////////
inline bool
sdkCheckErrorGL(const char *file, const int line)
{
    bool ret_val = true;

    // check for error
    GLenum gl_error = glGetError();

    if (gl_error != GL_NO_ERROR)
    {
#if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        char tmpStr[512];
        // NOTE: "%s(%i) : " allows Visual Studio to directly jump to the file at the right line
        // when the user double clicks on the error line in the Output pane. Like any compile error.
        sprintf_s(tmpStr, 255, "\n%s(%i) : GL Error : %s\n\n", file, line, gluErrorString(gl_error));
        fprintf(stderr, "%s", tmpStr);
#endif
        fprintf(stderr, "GL Error in file '%s' in line %d :\n", file, line);
        fprintf(stderr, "%s\n", gluErrorString(gl_error));
        ret_val = false;
    }

    return ret_val;
}

#define SDK_CHECK_ERROR_GL()                                              \
    if( false == sdkCheckErrorGL( __FILE__, __LINE__)) {                  \
        DEVICE_RESET                                                      \
        exit(EXIT_FAILURE);                                               \
    }
#endif

#endif
