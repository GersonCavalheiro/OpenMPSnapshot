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
#define MAX_DATA  (1<<15)     
#define THRESHOLD 2000
int    more_data_arriving( int );
int    getting_data( int ** );
double heavy_work( int );
int main ( int argc, char **argv )
{
srand48(time(NULL));
int  Nthreads;
int  iteration = 0;
int  data_are_arriving;
int  ndata;
int  bunch;
int  next_bunch = 0;
int *data; 
bunch = (argc > 1 ? atoi(*(argv+1)) : 10 );
#pragma omp parallel firstprivate(bunch)
{
int me = omp_get_thread_num();
#pragma omp single
{
Nthreads = omp_get_num_threads();
data_are_arriving = more_data_arriving(0);
}
while( data_are_arriving )
{
#pragma omp single                               
{
ndata = getting_data( &data );
printf("iteration %d: thread %d got %d data\n",
iteration, me, ndata );
}
int mystart;
do
{
int mystop;
#pragma omp atomic capture
{ mystart = next_bunch; next_bunch += bunch; }
if( mystart < ndata ) {
mystop = mystart + bunch;
mystop = (mystop > ndata ? ndata : mystop);
#if defined(DETAILS)
printf("\tthread %d processing [%d:%d]\n",
me, mystart, mystop);
#endif
for( ; mystart < mystop; mystart++ )
heavy_work( mystart ); }
} while( mystart < ndata );
#pragma omp barrier		             
#pragma omp single     	                     
{
while( nthreads_that_finished < nthreads); 
free( data );
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
int getting_data( int **data )
{
#define MIN  10
#define MAX 25
int howmany = lrand48() % MAX_DATA;
howmany = ( howmany == 0 ? 1 : howmany);
*data = (int*)calloc( howmany, sizeof(int));
for( int j = 0; j < howmany; j++ )
(*data)[j] = 1024 + lrand48() % (MAX-MIN);  
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
