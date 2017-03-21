#include "common.h"

void dinit_rand(double *vec, uint32_t len)
{
    uint32_t i;
    for (i = 0; i < len; i++)
        vec[i] = rand() / (double)RAND_MAX;
}

void finit_rand(float *vec, uint32_t len)
{
    uint32_t i;
    for (i = 0; i < len; i++)
        vec[i] = rand() / (float)RAND_MAX;
}

void iinit_rand(int *vec, uint32_t len)
{
    uint32_t i;
    for (i = 0; i < len; i++)
        vec[i] = rand();
}

int32_t fcheck_rel(float *A, float *B, uint32_t N, float th)
{
    int i;
    for (i = 0; i < N; i++) {
        if (ABS(A[i] - B[i]) / ABS(A[i]) > th)
        return 1;
    }
    return 0;
}

int32_t fcheck(float *A, float *B, uint32_t N, float th)
{
    int i;
    for (i = 0; i < N; i++) {
        if (ABS(A[i] - B[i]) > th) {
            printf("Test %d out of %d FAILED, %f %f\n", i, N, A[i], B[i]);
            return 1;
        }
    }
    return 0;
}

int32_t dcheck(double *A, double *B, uint32_t N, double th)
{
    int i;
    for (i = 0; i < N; i++) {
        if (ABS(A[i] - B[i]) > th)
        return 1;
    }
    return 0;
}

int32_t icheck(int *A, int *B, uint32_t N, int th)
{
    int i;
    for (i = 0; i < N; i++) {
        if (ABS(A[i] - B[i]) > th)
        return 1;
    }
    return 0;
}

float calc_bandwidth(int32_t memsize, double tu)
{
    float mbs = ((float)(1 << 10) * memsize) / (tu * (float)(1 << 20));
    return mbs;
}

void dump(float *arr, int size)
{
    int i;
    for (i = 0; i < size; i++) {
        printf("%f, ", arr[i]);
    }
        printf("\n");
}

int fread_float(const char *fn, float *data, int count)
{
    FILE *fl = fopen(fn, "r");
    if (!fl)
        return -1;
    int i = 0;
    for (i = 0; i < count; i++) {
        if (fscanf(fl, "%f", data + i) != 1)
        break;
    }
    fclose(fl);
    return i;
}

int fread_int(const char *fn, int *data, int count)
{
    FILE *fl = fopen(fn, "r");
    if (!fl)
        return -1;
    int i = 0;
    for (i = 0; i < count; i++) {
        if (fscanf(fl, "%d", data + i) != 1)
        break;
    }
    fclose(fl);
    return i;
}
