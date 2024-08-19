#pragma GCC optimize("O3","unroll-loops","omit-frame-pointer","inline", "unsafe-math-optimizations")
#pragma GCC option("arch=native","tune=native","no-zero-upper")
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#define SEED 1053608
#define N 1000000000
int main()
{
int count=0; 
double pi;
srand(SEED);
for (int i=0; i<N; i++) {
double x,y;
x = (double)rand()/RAND_MAX;
y = (double)rand()/RAND_MAX;
if (x*x+y*y <= 1)
count++;
}
pi=(double)count/N*4;
printf("Single : # of trials = %14ld , estimate of pi is %1.16f AND an absolute error of %g\n",N,pi,fabs(pi - M_PI));
return 0;
}