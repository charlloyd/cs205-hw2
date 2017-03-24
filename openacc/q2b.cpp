#include "helper.cpp"
#include <stdarg.h>

void matrix_multiply(int *a, int *b, int *c, int nrows_a, int ncols_a, int ncols_b)
{
   #pragma acc enter data copyin(a[0:nrows_a * ncols_a],b[0:ncols_a * ncols_b])

   #pragma acc kernels loop independent
   for(int i=0; i<nrows_a; i++)
   {
    #pragma acc kernels loop independent
     for(int j=0; j < ncols_b; j++)
      {
        c[i*ncols_b+j] = 0;

        #pragma acc loop independent reduction(+:sum)
        for(int k=0;k < ncols_a; k++)
        {
           c[i*ncols_b+j] += a[i*ncols_a+k] * b[k*ncols_b+j];
        }
      }
   }

  #pragma acc exit data copyout(C[0:nrows_a * ncols_b])

}

int main(void)
{

  matrix A = matrix(4,3);
  matrix B = matrix(3,4);
  matrix C = matrix(4,4);

  A.set_to_random_int();
  B.set_to_random_int();

  matrix_multiply(A.data,B.data,C.data,A.nrows,A.ncols,B.ncols);

  A.print();
  std::cout<<std::endl;
  B.print();
  std::cout<<std::endl;
  C.print();

  int problem_sizes[3] = {1<<3, 1<<5, 1<<8}; //sqrt of N

  for(int k=0; k<0; k++)
  {
     int N = problem_sizes[k];

     matrix A = matrix(N,N);
     matrix B = matrix(N,N);
     matrix C = matrix(N,N);

     A.set_to_random_int();
     B.set_to_random_int();

     clock_t begin = clock();
     matrix_multiply(A.data,B.data,C.data,A.nrows,A.ncols,B.ncols);
     clock_t end = clock();
     int time_spent = (int)(end - begin) / CLOCKS_PER_SEC;

     std::cout<<"problem size: "<<N*N<<" time:"<< time_spent << std::endl;

  }


}

