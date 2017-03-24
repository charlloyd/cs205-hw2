#include <iostream>
#include <stdlib.h>
#include <stdarg.h>

void  copy_from_source_to_destination(int *x, int *y, int n)
{
   for(int k=0;k<n;k++) y[k] = x[k];
}


int* generate_graph(nrows,ncols)
{
   data = (int *) malloc(sizeof(int) * n * n);
   for(int i=0;i < nrows*ncols;++i) data[i] = rand()%2;
   return data
}



int min(int i, int j)
{
   if (i < j) return i;
   return j;
}


void print(int *data,int nrows,int ncols)
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




bool same(int *x, int *y, int n)
{
   

   for(int k=0;k<n;++k)
   {
     if (y[k] != x[k]) return false;
     
   }
   return true;
}
