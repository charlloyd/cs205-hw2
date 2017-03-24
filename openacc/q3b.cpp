#include "helper.cpp"
#include <algorithm>
#include <limits.h>

void apsp(int *A0, int n)
{

   int *Anp1 = (int *) malloc(sizeof(int) * n * n);
   int *An   = (int *) malloc(sizeof(int) * n * n);

   copy_from_source_to_destination(A0, Anp1, n*n);
   copy_from_source_to_destination(A0, An  , n*n);

   while(true)
   {
      for(int i=0; i<n; i++)
      {
         for(int j=0; j < n; j++)
         {
           Anp1[i*n+j] = An[i*n+j];

           for(int k=0;k < n; k++) Anp1[i*n+j] = std::min(Anp1[i*n+j], An[i*n+k] + A0[k*n+j]);
         }
      }

      print(Anp1,n,n);

      if (same(Anp1,An,n*n)) break;
      copy_from_source_to_destination(Anp1, An  , n*n);

   }
}

int main(void)
{
  int data[16] = {0,INT_MAX/2,-2,INT_MAX/2,4,0,3,INT_MAX/2,INT_MAX/2,INT_MAX/2,0,2,INT_MAX/2,-1,INT_MAX/2,0};
  apsp(data, 4);

}
