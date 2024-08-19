#include <stdio.h>
#include <stdlib.h>
#define N 100
struct myInt {
int x;
};
#pragma omp declare reduction(+:struct myInt: omp_out.x += omp_in.x) initializer(omp_priv = {0})
int omp_get_num_threads(void);
int omp_get_thread_num(void);
int main (int argc, char **argv)
{
int i,s=0;
int a[N];
struct myInt x = {0};
for ( i = 0; i < N ; i++ ) {
a[i] = i;
s += i;
}
#pragma omp parallel for reduction(+:x)
for ( i = 0; i < N ; i++ )
{
x.x += a[i];
}
if ( x.x != s ) abort();
return 0;
}
