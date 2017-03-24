#include "helper.cpp"
#include <stdarg.h>

void matrix_multiply(int *a, int *b, int *c, int nrows_a, int ncols_a, int ncols_b)
{
  #pragma acc enter data copyin(a[0:nrows_a * ncols_a],b[0:ncols_a * ncols_b])
  #pragma acc enter data create(c[0:nrows_a * ncols_b])
  #pragma acc data present(a[0:nrows_a * ncols_a], b[0:ncols_a * ncols_b], c[0:nrows_a * ncols_b])
    
   #pragma acc kernels loop independent
   for(int i=0; i<nrows_a; i++)
   {
     #pragma acc loop independent
     for(int j=0; j < ncols_b; j++)
      {
          float sum = 0;
          
        
        #pragma acc loop independent reduction(+:sum)
        for(int k=0;k < ncols_a; k++)
        {
            float A = a[i*ncols_a+k];
            float B = b[k*ncols_b+j];
            sum += A * B;
        }
          c[i*ncols_b+j] = sum;
      }
   }
    #pragma acc exit data copyout(c[0:nrows_a * ncols_b])
    #pragma acc exit data delete(a[0:nrows_a * ncols_a], b[0:ncols_a * ncols_b])

}

int main(void)
{

    int problem_sizes[2] = {2<<6, 2<<10};//, 1<<8}; //sqrt of N
    printf("%d", 0);
  for(int k=0; k<2; k++) {
      printf("%d", 1);
     int N = problem_sizes[k];
      printf("%d", 2);
     int *A = generate_graph(N,N);
     int *B = generate_graph(N,N);
     int *C = generate_graph(N,N);
      
      printf("%d", 3);
     clock_t begin = clock();
     matrix_multiply(A,B,C,N,N,N);
     clock_t end = clock();
     int time_spent = (int)(end - begin) / CLOCKS_PER_SEC;
      
    printf("%d", 4);
     std::cout<<"problem size: "<<N*N<<" time:"<< time_spent << std::endl;
    printf("%d", 5);
  }


}

