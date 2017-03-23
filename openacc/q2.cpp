#include <iostream>
#include <random>

class matrix
{
  public:
    matrix(int n);
    matrix(int n, int m);
    matrix(int *data, int n, int m);
    void set_to_random_int();
    void print();
    void set_to_ones();
    void generate_graph();
    bool same(matrix &B);

    int ncols;
    int nrows;
    int *data;

};

bool matrix::same(matrix &B)
{
   int N = nrows * ncols;

   for(int i=0;i<N;++i)
   {
      if (data[i] != B.data[i]) return false;
   }
   return true;
}

matrix::matrix(int *D, int n, int m)
{
  ncols = n;
  nrows = n;
  data = (int *) malloc( sizeof(int)*n*m);


}

matrix::matrix(int n)
{
  ncols = n;
  nrows = n;
  data = (int *) malloc( sizeof(int) * n * n);
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
  data = (int *) malloc( sizeof(int) * n * m);
}

void matrix::set_to_random_int()
{
   std::mt19937 engine;
   engine.seed(std::random_device()());
   std::uniform_int_distribution<> dis(-1,1);

   for(int i=0;i < nrows*ncols;++i) data[i] = dis(engine);
}

void matrix::generate_graph()
{
   std::mt19937 engine;
   engine.seed(std::random_device()());
   std::uniform_int_distribution<> dis(0,1);

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


void matrix_multiply(int *a, int *b, int *c, int nrows_a, int ncols_a, int ncols_b)
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

  #pragma acc exit data copyout(c[0:nrows_a * ncols_b])

}


bool same(int *x, int *y, int n)
{
   for(int k=0;k<n;++k)
   {
     if (y[k] != x[k]) return false;
   }
   return true;
}



void bfs(int *A, int *x, int n, int root)
{
   #pragma acc enter data copyin(A[0:n * n],x[0:n])

   int *last_x = (int *)malloc(sizeof(int) * n);

   for(int i=0;i < n;i++)
   {
     x[i]=0; last_x[i]=0;
   }
   x[root]=1; last_x[root]=1;

   bool converged = false;

   while(true)
   {
     copy_from_source_to_destination(x, last_x, n);

     for(int j=0; j < n; j++)
     {
        x[j] = 0;

        for(int k=0;k < n; k++)
        {
           x[j] = x[j] || (A[j*n+k] && last_x[k]);
        }
     }

     for(int ii=0;ii<n;++ii) std::cout<<x[ii]<<" "; std::cout<<std::endl;
     for(int ii=0;ii<n;++ii) std::cout<<last_x[ii]<<" "; std::cout<<std::endl;

     if (same(x,last_x,n)) break;

   }

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

  N=3;
  matrix D = matrix(N);
  matrix X = matrix(N,1);
  D.generate_graph();

  //D.print();

  bfs(D.data, X.data, N, 1);

}
