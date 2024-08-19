#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations")
#pragma GCC option("arch=native","tune=native","no-zero-upper")
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <limits.h>
#include <omp.h>
#define SEED 1053608
#define N 5000000000000
unsigned int seed = 676767676 ; 
double randUint( long i ){
seed = seed * 1103515245 + 123456;
return seed / (double)UINT_MAX ;
}
int main()
{
long count=0;
double pi;
#pragma omp parallel for reduction(+: count)
for (long i=0; i<N; i++) {
double x,y;
x = randUint(i);
y = randUint(i);
if (x*x+y*y <= 1)
count = count + 1;
}
pi=((double)count/(double)N) * 4.0;
printf("OpenMP : # of trials = %14ld , estimate of pi is %1.16f AND an absolute error of %g\n",N,pi,fabs(pi - M_PI));
return 0;
}