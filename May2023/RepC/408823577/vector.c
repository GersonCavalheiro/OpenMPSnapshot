#define _XOPEN_SOURCE 700
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>
#include <papi.h>
#include "mypapi.h"
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define ALIGN 64
#if defined(__AVX512F__) || defined(__AVX512__)
#undef ALIGN
#define ALIGN 128
#define V_DSIZE 8
typedef __m512d vd;
vd _dzero_ = {0, 0, 0, 0, 0, 0, 0, 0};
#define MUL_ADD( A1, A2, R ) _mm512_fmadd_pd( (A1), (A2), (R) )
#elif defined ( __AVX__ ) || defined ( __AVX2__ )
#define V_DSIZE 4
typedef __m256d vd;
vd _dzero_ = {0, 0, 0, 0};
#define MUL_ADD( A1, A2, R ) _mm256_fmadd_pd( (A1), (A2), (R) )
#elif defined ( __SSE4__ ) || defined ( __SSE2__ )
#define V_DSIZE 2
typedef __m128d vd;
vd _dzero_ = {0, 0};
#define MUL_ADD( A1, A2, R ) _mm128_fmadd_pd( (A1), (A2), (R) )
#else
#define V_DSIZE 1
typedef double vd;
vd _zero_ = 0;
#define MUL_ADD( A1, A2, R ) (R) += (A1) * (A2)
#endif
typedef union {
vd   V;
double v[V_DSIZE];
}vd_u;
#define _DO_NOT_OPTIMIZE_BEGIN asm volatile("" ::: "memory");
#define _DO_NOT_OPTIMIZE_END asm volatile("" ::: "memory");
#ifndef _MEM_CLOCK
#define _MEM_CLOCK 1867            
#endif
#ifndef _MEM_WIDTH
#define _MEM_WIDTH 64              
#endif
int main(int argc, char **argv)
{
double * restrict array1 __attribute__((aligned(ALIGN)));
double * restrict array2 __attribute__((aligned(ALIGN)));
double tstart, tend;
int    N;
struct timespec ts;
if(argc > 1)
N = atoi(*(argv+1));
else
N = 1000000;
array1 = (double*)aligned_alloc(ALIGN, N * sizeof(double));
array2 = (double*)aligned_alloc(ALIGN, N * sizeof(double));
srand48(12983476);
printf("generating %d numbers..", 2*N); fflush(stdout);
tstart = CPU_TIME;
{
int i;
i = N;
while ( i )
array1[--i] = drand48();
i = N;
while ( i )
array2[--i] = drand48();
}
tend = CPU_TIME - tstart;
printf("done (in %6.3gsec)\n", tend);
PAPI_INIT;
#define ITER 10
double std_dev =0,  avg_timing = 0, min_time = N;
double sum;
double tstart_all = CPU_TIME;
PAPI_START_CNTR;
for(int k = 0; k < ITER; k++)
{
sum = 0;
vd sum_ __attribute__ ((aligned(64)));
sum_ = _dzero_;
vd * restrict A1 __attribute__ ((aligned(64)));
vd * restrict A2 __attribute__ ((aligned(64)));
A1  = (vd*)array1 ;
A2  = (vd*)array2 ;
int N__ = N / V_DSIZE;
int N_  = N__ * V_DSIZE;
tstart = CPU_TIME;
#pragma ivdep
for( int i = 0; i < N__; i++)
sum_ = MUL_ADD( A1[i], A2[i], sum_ );
for ( int i = N_; i < N; i++ )
sum += array1[ i ] * array2[ i ];     
vd_u sum_u __attribute__ ((aligned(64)));
sum_u.V = sum_;
sum_u.v[0] = sum_u.v[0] + sum_u.v[1];
sum_u.v[2] = sum_u.v[2] + sum_u.v[3];
sum += sum_u.v[0] + sum_u.v[2];
tend = CPU_TIME - tstart;
avg_timing += tend;
std_dev    += tend * tend;
if ( tend < min_time )
min_time = tend;
}
double tend_all = CPU_TIME - tstart_all;
PAPI_STOP_CNTR;
avg_timing /= ITER;
std_dev = sqrt(std_dev / ITER - avg_timing*avg_timing);    
printf("sum is %g\ntime is: %g (min %g, std_dev %g, all %g)\n",
sum, avg_timing, min_time, std_dev, tend_all);
double max_GB_per_sec          = (double)_MEM_CLOCK / 1000 * (_MEM_WIDTH / 8);
double transfer_rate_in_GB_sec = (double)N*2*sizeof(double) / (1024*1024*1024) / avg_timing;  
printf("transfer rate was %6.3g GB/sec (%7.4g%% of theoretical max that is %5.2g GB/sec)\n",
transfer_rate_in_GB_sec, transfer_rate_in_GB_sec / max_GB_per_sec * 100, max_GB_per_sec);
#ifdef USE_PAPI
printf( "\tIPC: %4.2g\n"
"\t[ time-per-element: %6.4gsec  -  cycles-per-element: %9.6g ]\n",
(double)papi_values[0] / papi_values[1],
avg_timing / (ITER*N),
(double)papi_values[1] / (ITER*N) );
#endif
memset ( array1, 0, sizeof(double) * N );
memset ( array2, 0, sizeof(double) * N );
free( array2 );
free( array1 );
return 0;
}
