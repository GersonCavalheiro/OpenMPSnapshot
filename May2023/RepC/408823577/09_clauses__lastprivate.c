#if defined(__STDC__)
#  if (__STDC_VERSION__ >= 199901L)
#     define _XOPEN_SOURCE 700
#  endif
#endif
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#define CPU_TIME ({struct timespec ts; double t;			\
clock_gettime( CLOCK_THREAD_CPUTIME_ID, &ts ),	\
t = (double)ts.tv_sec +				\
(double)ts.tv_nsec * 1e-9; t;})
#define DEFAULT   100000
#define THRESHOLD 0.1
int  valid_data  ( void );
void receive_data( double *, double *);
int main( int argc, char **argv )
{
int    N = ( argc > 1 ? atoi(*(argv+1)) : DEFAULT);
double data = 0;
double last_time = CPU_TIME;
srand48(time(NULL));    
printf("before the parallel region last_time (at %p) is %g\n\n",
&last_time, last_time );
#pragma omp parallel 
{
int me = omp_get_thread_num();
printf("thread %d: last_time (at %p) is %g\n", me, &last_time, last_time);
#pragma omp barrier
#pragma omp master
printf("entering the for loop..\n");
#pragma omp for lastprivate( last_time, data )
for( int j = 0; j < N; j++ )
{	
if( valid_data() )
receive_data( &data, &last_time);
}
}
printf("\nthe last valid reception %g happened at %g\n", data, last_time);
return 0;
}
int valid_data( void )
{
return drand48() > THRESHOLD;
}
void receive_data ( double *d, double *t)
{  
*d = drand48();
*t = CPU_TIME;
}
