#include "helper.cpp"

void bfs(int *Adj, int *x, int n, int root)
{
   #pragma acc enter data copyin(Adj[0:n * n], x[0:n])

   int *last_x = (int *)malloc(sizeof(int) * n);

   for(int i=0;i < n; i++)
   {
     x[i] = 0; last_x[i] = 0;
   }
   x[root] = 1; last_x[root] = 1;

   while(true)
   {
     copy_from_source_to_destination(x, last_x, n);

    #pragma acc kernels loop independent
    for(int j=0; j < n; j++)
     {
        x[j] = 0;
        #pragma acc kernels loop independent
        for(int k=0;k < n; k++)
        {
           x[j] = x[j] || (Adj[j*n+k] && last_x[j]);
        }
     }

     for(int ii=0;ii<n;++ii) std::cout<<x[ii]<<" "; std::cout<<std::endl;
     for(int ii=0;ii<n;++ii) std::cout<<last_x[ii]<<" "; std::cout<<std::endl;

     if (same(x,last_x,n)) break;

   }
    #pragma acc exit data copyout(x[0:n])
    #pragma acc exit data delete(Adj[0:n * n], x[0:n])
}

int main(void)
{

  int N=6;
  matrix D = matrix(N);
  matrix X = matrix(N,1);
  D.generate_graph();

  //D.print();

  bfs(D.data, X.data, N, 1);

}
