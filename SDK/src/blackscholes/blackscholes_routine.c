/*
 * Copyright (c) 2016, NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <openacc.h>

#include "timer.h"

#ifdef FP64
typedef double real;
#define SQRT(x) sqrt((x))
#define EXP(x)  exp((x))
#define FABS(x) fabs((x))
#define LOG(x)  log((x))
#else
typedef float real;
#define SQRT(x) sqrtf((x))
#define EXP(x)  expf((x))
#define FABS(x) fabsf((x))
#define LOG(x) logf((x))
#endif

const float      RISKFREE = 0.02f;
const float    VOLATILITY = 0.30f;

///////////////////////////////////////////////////////////////////////////////
// Polynomial approximation of cumulative normal distribution function
///////////////////////////////////////////////////////////////////////////////
#pragma acc routine seq
real CND(real d)
{
    const real       A1 = (real)0.31938153;
    const real       A2 = (real)-0.356563782;
    const real       A3 = (real)1.781477937;
    const real       A4 = (real)-1.821255978;
    const real       A5 = (real)1.330274429;
    const real RSQRT2PI = (real)0.39894228040143267793994605993438;

    real
        K = (real)1.0 / ((real)1.0 + (real)0.2316419 * FABS(d));

    real
        cnd = RSQRT2PI * EXP(- (real)0.5 * d * d) * 
        (K * (A1 + K * (A2 + K * (A3 + K * (A4 + K * A5)))));

    if(d > 0)
        cnd = (real)1.0 - cnd;

    return cnd;
}

////////////////////////////////////////////////////////////////////////////////
// Process an array of optN options
////////////////////////////////////////////////////////////////////////////////
void BlackScholes(
    real * restrict callResult,
    real * restrict putResult,
    real * restrict stockPrice,
    real * restrict optionStrike,
    real * restrict optionYears,
    real Riskfree,
    real Volatility,
    int optN,
    int accelerate)
{
#pragma omp parallel for if (accelerate)
#pragma acc parallel loop if (accelerate)
    for(int opt = 0; opt < optN; opt++)
    {
        real S = stockPrice[opt];
        real X = optionStrike[opt];
        real T = optionYears[opt]; 
        real R = Riskfree, V = Volatility;

        real sqrtT = SQRT(T);
        real    d1 = (LOG(S / X) + (R + (real)0.5 * V * V) * T) / (V * sqrtT);
        real    d2 = d1 - V * sqrtT;
        real CNDD1 = CND(d1);
        real CNDD2 = CND(d2);

        //Calculate Call and Put simultaneously
        real expRT = EXP(- R * T);
        callResult[opt] = (real)(S * CNDD1 - X * expRT * CNDD2);
        putResult[opt] = (real)(X * expRT * ((real)1.0 - CNDD2) - S * ((real)1.0 - CNDD1));
    }
}

float RandFloat(float low, float high){
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}

int main(int argc, char **argv)
{
    int  OPT_N = 4000000;
    int OPT_SZ = OPT_N * sizeof(float);
    
    int iterations = 10;
    if (argc >= 2) iterations = atoi(argv[1]);
    
    real
        //Results calculated by CPU for reference
        *callResultCPU,
        *putResultCPU,
        //GPU results
        *callResultGPU,
        *putResultGPU,
        //CPU instance of input data
        *stockPrice,
        *optionStrike,
        *optionYears; 
        
    real delta, ref, sum_delta, sum_ref, max_delta, L1norm, gpuTime;
    
    printf("Initializing data...\n");
    
    callResultCPU   = (real *)malloc(OPT_SZ);
    putResultCPU    = (real *)malloc(OPT_SZ);
    callResultGPU   = (real *)malloc(OPT_SZ);
    putResultGPU    = (real *)malloc(OPT_SZ);
    stockPrice      = (real *)malloc(OPT_SZ);
    optionStrike    = (real *)malloc(OPT_SZ);
    optionYears     = (real *)malloc(OPT_SZ);
    
    srand(5347);
    //Generate options set
    for(int i = 0; i < OPT_N; i++){
        callResultCPU[i] = (real)0.0;
        putResultCPU[i]  = (real)-1.0;
        callResultGPU[i] = (real)0.0;
        putResultGPU[i]  = (real)-1.0;
        stockPrice[i]    = (real)RandFloat(5.0f, 30.0f);
        optionStrike[i]  = (real)RandFloat(1.0f, 100.0f);
        optionYears[i]   = (real)RandFloat(0.25f, 10.0f);
    }
    
#ifdef _OPENACC
    // run once outside timer to initialize/prime
    acc_init(acc_device_nvidia);
#endif
    
    BlackScholes(
        callResultGPU,
        putResultGPU,
        stockPrice,
        optionStrike,
        optionYears,
        RISKFREE,
        VOLATILITY,
        OPT_N,
        1
    );

    printf("Running Unaccelerated Version %d iterations...\n", iterations);

    StartTimer();
    for (int i = 0; i < iterations; i++)
    {
        BlackScholes(
            callResultCPU,
            putResultCPU,
            stockPrice,
            optionStrike,
            optionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
            0
        );
    }
    double ms = GetTimer() / iterations;
    
    printf("Running Accelerated Version %d iterations...\n", iterations);
    
    StartTimer();
    for (int i = 0; i < iterations; i++)
    {
        BlackScholes(
            callResultGPU,
            putResultGPU,
            stockPrice,
            optionStrike,
            optionYears,
            RISKFREE,
            VOLATILITY,
            OPT_N,
            1
        );
    }
    double msAccelerated = GetTimer() / iterations;   
    
    //Both call and put is calculated
    printf("Options count             : %i     \n", 2 * OPT_N);
    printf("Unaccelerated:\n");
    printf("\tBlackScholes() time    : %f msec\n", ms);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (ms * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (ms * 1E-3));
           
    printf("Accelerated:\n");
    printf("\tBlackScholes() time    : %f msec\n", msAccelerated);
    printf("\t%f GB/s, %f GOptions/s\n", 
           ((double)(5 * OPT_N * sizeof(float)) * 1E-9) / (msAccelerated * 1E-3),
           ((double)(2 * OPT_N) * 1E-9) / (msAccelerated * 1E-3));
    
    printf("Comparing the results...\n");
    
    //Calculate max absolute difference and L1 distance
    //between CPU and GPU results
    sum_delta = 0;
    sum_ref   = 0;
    max_delta = 0;
    for(int i = 0; i < OPT_N; i++){
        ref   = callResultCPU[i];
        delta = fabs(callResultCPU[i] - callResultGPU[i]);
        if(delta > max_delta) max_delta = delta;
        sum_delta += delta;
        sum_ref   += fabs(ref);
    }
    L1norm = sum_delta / sum_ref;
    printf("L1 norm: %E\n", L1norm);
    printf("Max absolute error: %E\n\n", max_delta);  

    if (max_delta > 2.0e-5) {
       printf("Test FAILED\n");
    } else {
       printf("Test PASSED\n");
    }
    
    free(callResultCPU);
    free(putResultCPU);
    free(callResultGPU);
    free(putResultGPU);
    free(stockPrice);
    free(optionStrike);
    free(optionYears);

    return 0;
}
