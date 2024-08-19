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
int me;
#pragma omp threadprivate(me)
void do_work( double[2], double * );
int main( int argc, char **argv )
{
long int t = time(NULL);
srand48(t);
int N = 10;  
#pragma omp parallel
me = omp_get_thread_num(); 
for ( int j = 0; j < N; j++ )
{      
double result = 0;
#pragma omp parallel 
{
double seed[2];
#pragma omp single copyprivate(seed)
seed[0] = drand48(), seed[1] = drand48();
seed[0] += sqrt((double)me);
double thread_result;
do_work( seed, &thread_result );
#pragma omp atomic update
result += thread_result;
}
printf("iter %d: result is %g\n", j, result);
}    
printf("\n\n");
srand48(t);
for ( int j = 0; j < N; j++ )
{      
double result = 0;
double seed[2];
#pragma omp parallel 
{
#pragma omp single 
seed[0] = drand48(), seed[1] = drand48();
double myseed[2] = {seed[0] + sqrt((double)me),
seed[1]};
double thread_result;
do_work( myseed, &thread_result );
#pragma omp atomic update
result += thread_result;
}
printf("iter %d: result is %g\n", j, result);
}    
return 0;
}
void do_work( double seed[2], double *res )
{
double tmp = ( seed[0] > 0 ? log10(seed[0]) : sin(seed[0]) );
*res = seed[0] * seed[1] * tmp;
return;
}
