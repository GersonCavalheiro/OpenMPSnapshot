#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#define THRESHOLD 2000
int    more_data_arriving( int );
int    getting_data( int, int * );
double heavy_work( int );
int main ( int argc, char **argv )
{
srand48(time(NULL));
#pragma omp parallel
{
while( data_are_arriving )       
{
}
}
return 0;
}
int more_data_arriving( int i )
{
double p = (double)(THRESHOLD - i) / THRESHOLD;
return (drand48() < p);
}
int getting_data( int n, int *data )
{
#define MIN  1000
#define MAX 10000
int howmany = lrand48() % n;
howmany = ( howmany == 0 ? 1 : howmany);
for( int j = 0; j < howmany; j++ )
data[j] = 1024 + lrand48() % (MAX-MIN);  
return howmany;
}
double heavy_work( int N )
{
double guess = 3.141572 / 3 * N;
for( int i = 0; i < N; i++ )
{
guess = exp( guess );
guess = sin( guess );
}
return guess;
}
