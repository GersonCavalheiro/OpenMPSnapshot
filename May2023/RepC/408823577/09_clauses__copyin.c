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
int me;
int golden_values[3];
#pragma omp threadprivate( me, golden_values )
void set_up_golden_values( void );
void do_work( int, unsigned long long *);
int main( int argc, char **argv )
{
srand48(time(NULL));
int N = 10;  
#pragma omp parallel
me = omp_get_thread_num();
for ( int j = 0; j < N; j++ )
{
set_up_golden_values( );
unsigned long long result;
#pragma omp parallel copyin(golden_values)
{
unsigned long long thread_result;
do_work( me, &thread_result );
#pragma omp atomic update
result += thread_result;
}
}    
for ( int j = 0; j < N; j++ )
{
set_up_golden_values( );
int local_golden[3];
for( int i = 0; i < 3; i++ )
local_golden[i] = golden_values[i];
unsigned long long result;
#pragma omp parallel
{
for( int i = 0; i < 3; i++ )
golden_values[i] = local_golden[i];
unsigned long long thread_result;
do_work( me, &thread_result );
#pragma omp atomic update
result += thread_result;
}
}    
return 0;
}
void set_up_golden_values( void )
{
for( int j = 0; j < 3; j++ )	 
golden_values[j] = lrand48();
return;
}
void do_work( int who, unsigned long long *res )
{
unsigned long long sum = 0;
for( int j = 0; j < 3; j++ )
golden_values[j] += who;
for( int j = who*100; j < who*200; j++ )
sum += (golden_values[0]*j + golden_values[1]) % golden_values[2];
*res = sum;
return;
}
