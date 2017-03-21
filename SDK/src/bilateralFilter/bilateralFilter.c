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
    Image bilateral filtering example

    This sample uses OpenACC directives to perform a simple bilateral filter 
	on an image and measures performance.

    Bilateral filter is an edge-preserving nonlinear smoothing filter. There 
    are three parameters distribute to the filter: gaussian delta, euclidean 
    delta and iterations.

    When the euclidean delta increases, most of the fine texture will be
    filtered away, yet all contours are as crisp as in the original image.
    If the euclidean delta approximates to ∞, the filter becomes a normal
    gaussian filter. Fine texture will blur more with larger gaussian delta.
    Multiple iterations have the effect of flattening the colors in an 
    image considerably, but without blurring edges, which produces a cartoon
    effect.

    To learn more details about this filter, please view C. Tomasi's "Bilateral
    Filtering for Gray and Color Images".

*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "timer.h"

#define KERNEL_RADIUS 8
#define KERNEL_LENGTH (2 * KERNEL_RADIUS + 1)

void updateGaussian(float *h_Gaussian, float delta, int radius) {
    for (int i = 0; i < 2 * radius + 1; ++i) {
        int x = i - radius;
        h_Gaussian[i] = exp(-(x * x) / (2 * delta * delta));
    }
}

float heuclideanLen(float x1, float y1, float z1, float w1,
                    float x2, float y2, float z2, float w2,
                    float d) {
    float mod = (x2 - x1) * (x2 - x1) +
                (y2 - y1) * (y2 - y1) +
                (z2 - z1) * (z2 - z1) +
                (w2 - w1) * (w2 - w1);

    return exp(-mod / (2 * d * d));
}

#define RGBA_FLOAT2INT(x, y, z, w, c)                                    \
{                                                                        \
    unsigned int uW = (((unsigned int)(fabs(w) * 255.0f)) & 0xff) << 24; \
    unsigned int uZ = (((unsigned int)(fabs(z) * 255.0f)) & 0xff) << 16; \
    unsigned int uY = (((unsigned int)(fabs(y) * 255.0f)) & 0xff) << 8;  \
    unsigned int uX = ((unsigned int)(fabs(x) * 255.0f)) & 0xff;         \
    (c) = (uW | uZ | uY | uX);                                           \
}

#define RGBA_INT2FLOAT(c, x, y, z, w)                                    \
    (x) = ((c) & 0xff) * 0.003921568627f;         /*  /255.0f */         \
    (y) = (((c) >> 8) & 0xff) * 0.003921568627f;  /*  /255.0f */         \
    (z) = (((c) >> 16) & 0xff) * 0.003921568627f; /*  /255.0f */         \
    (w) = (((c) >> 24) & 0xff) * 0.003921568627f; /*  /255.0f */
    
double RGBAL2Dist(unsigned int a, unsigned int b) {
    float x1, y1, z1, w1, x2, y2, z2, w2;
    
    RGBA_INT2FLOAT(a, x1, y1, z1, w1)
    RGBA_INT2FLOAT(b, x2, y2, z2, w2)
    
    double deltaX = x2 - x1;
    double deltaY = y2 - y1;
    double deltaZ = z2 - z1;
    double deltaW = w2 - w1;
    
    return deltaX * deltaX + deltaY * deltaY + 
           deltaZ * deltaZ + deltaW * deltaW;
}

double RGBAL2Norm(unsigned int a) {
    float x, y, z, w;
    
    RGBA_INT2FLOAT(a, x, y, z, w)
    
    return x * x + y * y + z * z + w * w;
}

void bilateralFilter(unsigned int * restrict h_Src, 
                     unsigned int * restrict h_Dst,
                     float * restrict h_Gaussian,
                     float e_d,
                     int imageW, 
                     int imageH, 
                     int kernelR,
                     int accelerate) {
    float domainDist, colorDist, factor;
    
    float * restrict h_BufferX = 
        (float *) malloc(imageW * imageH * sizeof(float));
    float * restrict h_BufferY = 
        (float *) malloc(imageW * imageH * sizeof(float));
    float * restrict h_BufferZ = 
        (float *) malloc(imageW * imageH * sizeof(float));
    float * restrict h_BufferW = 
        (float *) malloc(imageW * imageH * sizeof(float));
    
#pragma acc kernels                                                \
        copyin(h_Src[0:imageW*imageH],h_Gaussian[0:KERNEL_LENGTH]) \
        copyout(h_Dst[0:imageW*imageH])                            \
        create(h_BufferX[0:imageW*imageH],                         \
               h_BufferY[0:imageW*imageH],                         \
               h_BufferZ[0:imageW*imageH],                         \
               h_BufferW[0:imageW*imageH])                         \
        if(accelerate)
    {
#pragma acc for independent, parallel
        for (int y = 0; y < imageH; ++y) {
#pragma acc for independent, parallel
            for (int x = 0; x < imageW; ++x) {
                RGBA_INT2FLOAT(h_Src[y * imageW + x], 
                            h_BufferX[y * imageW + x],
                            h_BufferY[y * imageW + x],
                            h_BufferZ[y * imageW + x],
                            h_BufferW[y * imageW + x])
            }
        }

#pragma acc for independent, parallel
        for (int y = 0; y < imageH; ++y) {
#pragma acc for independent, parallel
            for (int x = 0; x < imageW; ++x) {
                float tX = 0.0f, tY = 0.0f, tZ = 0.0f, tW = 0.0f;
                float sum = 0.0f;

                for (int i = -kernelR; i <= kernelR; ++i) {
                    int neighborY = y + i;

                    //clamp the neighbor pixel, prevent overflow
                    if (neighborY < 0) {
                        neighborY = 0;
                    } else if (neighborY >= imageH) {
                        neighborY = imageH - 1;
                    }

                    for(int j = -kernelR; j <= kernelR; ++j) {
                        domainDist = 
                            h_Gaussian[kernelR + i] * h_Gaussian[kernelR + j];

                        //clamp the neighbor pixel, prevent overflow
                        int neighborX = x + j;
                        
                        if (neighborX < 0) {
                            neighborX = 0;
                        } else if(neighborX >= imageW) {
                            neighborX = imageW - 1;
                        }

                        colorDist = 
                            heuclideanLen(
                                h_BufferX[neighborY * imageW + neighborX],
                                h_BufferY[neighborY * imageW + neighborX],
                                h_BufferZ[neighborY * imageW + neighborX],
                                h_BufferW[neighborY * imageW + neighborX],
                                h_BufferX[y * imageW + x],
                                h_BufferY[y * imageW + x], 
                                h_BufferZ[y * imageW + x], 
                                h_BufferW[y * imageW + x], 
                                e_d);
                            
                        factor = domainDist * colorDist;
                        sum += factor;
                        
                        tX += 
                            factor * h_BufferX[neighborY * imageW + neighborX];
                        tY += 
                            factor * h_BufferY[neighborY * imageW + neighborX];
                        tZ += 
                            factor * h_BufferZ[neighborY * imageW + neighborX];
                        tW += 
                            factor * h_BufferW[neighborY * imageW + neighborX];
                    }
                }
                
                tX /= sum;
                tY /= sum;
                tZ /= sum;
                tW /= sum;
                
                RGBA_FLOAT2INT(tX, tY, tZ, tW, h_Dst[y * imageW + x])
            }
        }
    }
    
    free(h_BufferX);
    free(h_BufferY);
    free(h_BufferZ);
    free(h_BufferW);
}

// Main program
int main( int argc, char** argv) { 
    int totalIterations = 2;
    
    if (argc >= 2) {
        totalIterations = atoi(argv[1]);
    }
    
    float
        *h_Kernel;
        
    unsigned int
        *h_Input,
        *h_Buffer,
        *h_OutputAcc,
        *h_OutputNonAcc;

    const int imageW = 3072;
    const int imageH = 3072;
    
    const float gaussianDelta = 4;
    const float euclideanDelta = 0.1f;
    const int filterRadius = 5;
    
    printf("Image Width x Height = %i x %i\n", imageW, imageH);
    printf("* Allocating and initializing host arrays...\n");
    
    h_Kernel = 
        (float *) malloc(KERNEL_LENGTH * sizeof(float));
    h_Input = 
        (unsigned int *) malloc(imageW * imageH * sizeof(unsigned int));
    h_OutputAcc = 
        (unsigned int *) malloc(imageW * imageH * sizeof(unsigned int));
    h_OutputNonAcc = 
        (unsigned int *) malloc(imageW * imageH * sizeof(unsigned int));
        
    srand(200);
    
    updateGaussian(h_Kernel, gaussianDelta, filterRadius);
    
    for(unsigned i = 0; i < imageW * imageH; i++) {
        unsigned int color = rand() % 256;
        color = (color << 8) | (rand() % 256);
        color = (color << 8) | (rand() % 256);
        
        h_Input[i] = color;
    }
    
    // Warm up iteration
    {
        bilateralFilter(h_Input, 
                        h_OutputAcc, 
                        h_Kernel, 
                        euclideanDelta, 
                        imageW,
                        imageH,
                        filterRadius,
                        1);
    }
    
    printf("* Running accelerated version (%u iterations)...\n", 
           totalIterations);
    StartTimer();
    
    for (int iteration = 0; iteration < totalIterations; ++iteration) {
        bilateralFilter(h_Input, 
                        h_OutputAcc, 
                        h_Kernel, 
                        euclideanDelta, 
                        imageW,
                        imageH,
                        filterRadius,
                        1);
    }
    
    double accMs = GetTimer();
    
    printf("* Running non-accelerated version (%u iterations)...\n",
           totalIterations);
    StartTimer();
    
    for (int iteration = 0; iteration < totalIterations; ++iteration) {
        bilateralFilter(h_Input, 
                        h_OutputNonAcc, 
                        h_Kernel, 
                        euclideanDelta, 
                        imageW,
                        imageH,
                        filterRadius,
                        0);
    }
    
    double nonAccMs = GetTimer();
    
    printf("* Calculating relative L2 norm...\n");
    
    double deltaL2Norm = 0;
    double nonAccL2Norm = 0;
    for (unsigned int i = 0; i < imageW * imageH; ++i) {
        deltaL2Norm += RGBAL2Dist(h_OutputAcc[i], h_OutputNonAcc[i]);
        nonAccL2Norm += RGBAL2Norm(h_OutputNonAcc[i]);
    }
    
    double L2Norm = sqrt(deltaL2Norm / nonAccL2Norm);
    
    printf("***** Summary *****\n");
    printf("Accelerated version finished in %f ms;\n", accMs);
    printf("Non-accelerated version finished in %f ms;\n", nonAccMs);
    printf("Speedup is %f times;\n", nonAccMs / accMs);
    printf("Relative L2 norm is %E.\n", L2Norm);

    if (L2Norm > 5.0e-2) {
       printf("Test FAILED\n");
    } else {
       printf("Test PASSES\n");
    }

    free(h_OutputAcc);
    free(h_OutputNonAcc);
    free(h_Input);
    free(h_Kernel);

    return 0;
}
