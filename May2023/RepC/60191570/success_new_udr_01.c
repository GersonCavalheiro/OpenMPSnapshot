#include <stdio.h>
#include <stdlib.h>
#pragma omp declare reduction(mymin : int : omp_out = omp_out > omp_in ? omp_in : omp_out ) initializer(omp_priv = 2147483647)
#define N 100
int omp_get_num_threads(void);
int omp_get_thread_num(void);
int main (int argc, char **argv)
{
int i,x = N+1;
int a[N];
for ( i = 0; i < N ; i++ ) a[i] = i;
#ifdef NANOX
#pragma omp for reduction(mymin:x)
#else
#pragma omp parallel for reduction(mymin:x)
#endif
for ( i = 0; i < N ; i++ )
{
x = a[i] < x ? a[i] : x;
}
if ( x != 0 )
abort();
return 0;
}
