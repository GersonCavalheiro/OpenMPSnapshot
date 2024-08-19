#include "wtime.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
void wtime( double * );
static double elapsed_time( void )
{
double t;
#if defined(_OPENMP) && (_OPENMP > 200010)
t = omp_get_wtime();
#else
wtime( &t );
#endif
return( t );
}
static double start[64], elapsed[64];
static unsigned count[64];
#ifdef _OPENMP
#pragma omp threadprivate(start, elapsed, count)
#endif
void timer_clear( int n )
{
elapsed[n] = 0.0;
count[n] = 0;
}
void timer_start( int n )
{
start[n] = elapsed_time();
}
void timer_stop( int n )
{
double t, now;
now = elapsed_time();
t = now - start[n];
elapsed[n] += t;
count[n]++;
}
double timer_read( int n )
{
return( elapsed[n] );
}
unsigned timer_count( int n )
{
return count[n];
}
