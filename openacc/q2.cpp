#include <iostream>
#include <random>

class matrix
{
  public:
    matrix(int n);
    matrix(int n, int m);
    void set_to_random_real();
    void set_to_random_int();
    void print();
    void set_to_ones();

    int ncols;
    int nrows;
    double *data;

};

matrix::matrix(int n)
{
  ncols = n;
  nrows = n;
  data = (double *) malloc( sizeof(double) * n * n);
}

int min(int i, int j)
{
   if (i < j) return i;
   return j;
}

int get_index(int i, int j, int ncols)
{
   return i*ncols+j;
}


void matrix::set_to_ones()
{
   int n = min(nrows,ncols);

   for(int i=0;i<n;i++) data[i*ncols+i] = i;
}

matrix::matrix(int n, int m)
{
  nrows = n;
  ncols = m;
  data = (double *) malloc( sizeof(double) * n * m);
}

void matrix::set_to_random_real()
{
   std::mt19937 engine;
   engine.seed(std::random_device()());
   std::uniform_real_distribution<> dis(-1,1);

   for(int i=0;i < nrows*ncols;++i) data[i] = dis(engine);
}

void matrix::set_to_random_int()
{
   std::mt19937 engine;
   engine.seed(std::random_device()());
   std::uniform_int_distribution<> dis(-1,1);

   for(int i=0;i < nrows*ncols;++i) data[i] = dis(engine);
}

void matrix::print()
{
   for(int i=0;i<nrows;++i)
   {
      for(int j=0;j<ncols;++j)
      {
        std::cout<< data[i*ncols+j]<<" ";
      }
      std::cout<<std::endl;
   }
}

bool mult_comp(matrix &a, matrix &b, matrix &c)
{
   return (a.ncols == b.nrows) & (a.nrows == c.nrows) & (b.ncols == c.ncols);
}


void matrix_multiply(double *a, double *b, double *c, int nrows_a, int ncols_a, int ncols_b)
{
   #pragma acc enter data copyin(a[0:nrows_a * ncols_a],B[0:ncols_a * ncols_b])

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

  int N;
  int problem_sizes[3] = {1<<3, 1<<5, 1<<8}; //sqrt of N

  for(int k=0; k<3; k++)
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
     double time_spent = (double)(end - begin) / CLOCKS_PER_SEC;

     std::cout<<"problem size: "<<N*N<<" time:"<< time_spent << std::endl;

  }


}
