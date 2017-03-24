#include "helper.cpp"
#include <stdarg.h>

void matrix_multiply(int *a, int *b, int *c, int nrows_a, int ncols_a, int ncols_b)
{

   for(int i=0; i<nrows_a; i++)
   {
   
     for(int j=0; j < ncols_b; j++)
      {
        c[i*ncols_b+j] = 0;

        for(int k=0;k < ncols_a; k++)
        {
           c[i*ncols_b+j] += a[i*ncols_a+k] * b[k*ncols_b+j];
        }
      }
   }

}

int main(void)
{

  int problem_sizes[3] = {1<<3, 1<<5, 1<<8}; //sqrt of N

  for(int k=0; k<0; k++)
  {
     int N = problem_sizes[k];

     int *A = generate_graph(N,N);
     int *B = generate_graph(N,N);
     int *C = generate_graph(N,N);

     clock_t begin = clock();
     matrix_multiply(A,B,C,N,N,N);
     clock_t end = clock();
     int time_spent = (int)(end - begin) / CLOCKS_PER_SEC;

     std::cout<<"problem size: "<<N*N<<" time:"<< time_spent << std::endl;

  }


}

