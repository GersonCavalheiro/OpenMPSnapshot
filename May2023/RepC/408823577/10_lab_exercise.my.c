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
int  Nthreads;
int  iteration = 0;
int  data_are_arriving;
int  ndata;
int *data; 
#pragma omp parallel
{
int me = omp_get_thread_num();
#pragma omp single
{
Nthreads = omp_get_num_threads();
data     = (int*)calloc(Nthreads, sizeof(int));   
data_are_arriving = more_data_arriving(0);
}
while( data_are_arriving )
{
#pragma omp single                               
{
ndata = getting_data( Nthreads, data );
printf("iteration %d: thread %d got %d data\n",
iteration, me, ndata );
}
if( me < ndata )                                
{                                             
heavy_work( data[me] );                     
}                                             
#pragma omp single     	                        
{                                               
if( !(data_are_arriving = more_data_arriving(iteration+1)) )
printf("\t>>> iteration %d : thread %d got the news that "
"no more data will arrive\n",
iteration, me);
else
iteration++;                                
}
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
