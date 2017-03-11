//
//  testCase.c
//  
//
//  Created by Eric Dunipace on 3/10/17.
//
//

#include "testCase.h"
#include <stdlib.h>
#include ""

void printArray(double *array, int size)
{
    for(int s = 0; s<size;++s){
        printf("%f, ",  array[s]);
    }
}


int main( int argc, char *argv[] )
{
    int SIZES[2] = {2, 4};//, 6, 8, 10};
    
    
    for(int s = 0; s < sizeof(SIZES)/sizeof(int); ++s){
        int size = SIZES[s];
        double testMat[size * size];
        double *out;
        double *blasOut;
        
        for(int t = 0; t < size*size; ++t){
            testMat[t] = t + 1.0;
        }
        
        size_t numbytes = size * size * sizeof( double );
        
        out = (double *) malloc ( numbytes );
        blasOut = (double *) malloc ( numbytes );
        
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
        printArray(out, size*size);
        printf("\n");
        printArray(testMat, size*size);
        printf("\n");
        printf("%d", s);
        free(out);
//        free(testMat);
    }
    
    
    return 0;
}
