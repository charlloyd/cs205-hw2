//
//  testCase.c
//  
//
//  Created by Eric Dunipace on 3/10/17.
//
//

#include "testCase.h"


void printArray(double *array, int size)
{
    for(int s = 0; s<size;++s){
        printf("%f, ",  array[s]);
    }
}


int main( int argc, char *argv[] )
{
    int SIZES[5] = {2, 4, 6, 8, 10};
    
    
    for(int s = 0; s < sizeof(SIZES)/sizeof(int); ++s){
        int size = SIZES[s];
        double testMat[size * size];
        double *out;
        double *blasOut;
        
        for(int t = 0; t < size*size; ++t){
            testMat[t] = t + 1.0;
        }
        
        out = (double *) calloc ( size * size, sizeof( double ) );
        blasOut = (double *) calloc ( size * size, sizeof( double )  );

        cpu_mm(size, size, size, testMat, testMat, out );
        
        cblas_dgemm(CblasColMajor,
                    CblasNoTrans,
                    CblasNoTrans,
                    size,
                    size,
                    size,
                    1.0,
                    testMat,
                    size,
                    testMat,
                    size,
                    1.0,
                    blasOut,
                    size);
        printf("%d", allEqual(out, blasOut, size*size, 1E-3));
        printf("\n");
        free(out);
        free(blasOut);
    }
    
    
    return 0;
}
