#include <iostream>
#include <stdlib.h>


void  copy_from_source_to_destination(int *x, int *y, int n)
{
   for(int k=0;k<n;k++) y[k] = x[k];
}

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
  nrows = m;
  data = (int *) malloc( sizeof(int)*n*m);
  copy_from_source_to_destination(D,data,n*m);
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
   for(int i=0;i < nrows*ncols;++i) data[i] = (rand()%3)-1;
}

void matrix::generate_graph()
{
   for(int i=0;i < nrows*ncols;++i) data[i] = rand()%2;
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
   std::cout<<std::endl;
}

bool mult_comp(matrix &a, matrix &b, matrix &c)
{
   return (a.ncols == b.nrows) & (a.nrows == c.nrows) & (b.ncols == c.ncols);
}


bool same(int *x, int *y, int n)
{
   

   for(int k=0;k<n;++k)
   {
     if (y[k] != x[k]) return false;
     
   }
   return true;
}
