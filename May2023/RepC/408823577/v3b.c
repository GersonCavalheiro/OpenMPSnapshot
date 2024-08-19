#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <math.h>
#ifdef USE_PAPI
#include <papi.h>
#define GET_IPC( RT, PT, I, IPC ) PAPI_ipc( (RT), (PT), (I), (IPC) )
#define ACC_CNTRS( RT1, PT1, I1, IPC1, RT2, PT2, I2, IPC2, RT, PT, I, IPC ) { \
RT  += RT2 - RT1;							\
PT  += PT2 - PT1;							\
I   += I2 - I1;							\
IPC += IPC2; }
#else
#define GET_IPC( RT, PT, I, IPC )
#define ACC_CNTRS( RT1, PT1, I1, IPC1, RT2, PT2, I2, IPC2, RT, PT, I, IPC )
#endif
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define DO_NOT_OPTIMIZE asm volatile("" ::: "memory");
#ifndef _MEM_CLOCK
#define _MEM_CLOCK 1867            
#endif
#ifndef _MEM_WIDTH
#define _MEM_WIDTH 64              
#endif
int main(int argc, char **argv)
{
double * restrict array1 __attribute__((aligned(32)));
double * restrict array2 __attribute__((aligned(32)));
double tstart, tend;
int    N;
struct timespec ts;
#ifdef USE_PAPI
float realtime1, proctime1, ipc1;
float realtime2, proctime2, ipc2;
float realtime, proctime, ipc;
long_long ins1, ins2, ins;
#endif
if(argc > 1)
N = atoi(*(argv+1));
else
N = 1000000;
array1 = (double*)aligned_alloc( 32, N * sizeof(double) );
array2 = (double*)aligned_alloc( 32, N * sizeof(double) );
srand48(12983476);
printf("generating %d numbers..", 2*N); fflush(stdout);
tstart = CPU_TIME;
{
for ( int i = 0; i < N; i++ )
array1[i] = drand48();
for ( int i = 0; i < N; i++ )
array2[i] = drand48();    
}
tend = CPU_TIME - tstart;
printf ( "done (in %6.3gsec)\n", tend );
#define ITER 10
double std_dev =0,  avg_time = 0, min_time = N;
double register sum;
double tstart_all = CPU_TIME;
GET_IPC( &realtime1, &proctime1, &ins1, &ipc1 );
for(int k = 0; k < ITER; k++)
{
sum = 0;
int N_4 = (N/4)*4;
double tstart = CPU_TIME;
double register a;
double register b;
a = array1[0] * array2[0];
#pragma ivdep      
for ( int i = 0; i < N_4; i+=4 )
{
DO_NOT_OPTIMIZE;                      
b = array1[i+4] * array2[i+4];
DO_NOT_OPTIMIZE;                      
sum += a +
array1[i+1] * array2[i+1] +
array1[i+2] * array2[i+2] +
array1[i+3] * array2[i+3];
a = b;
}
for ( int i = N_4; i < N; i++ )
sum += array1[i] * array2[i];
double time = CPU_TIME - tstart;
avg_time += time;
std_dev  += time * time;
if ( time < min_time )
min_time = time;
}
double tend_all = CPU_TIME - tstart_all;
GET_IPC( &realtime2, &proctime2, &ins2, &ipc2 );
ACC_CNTRS( realtime1, proctime1, ins1, ipc1,	   \
realtime2, proctime2, ins2, ipc2,	   \
realtime, proctime, ins, ipc );
avg_time /= ITER;
std_dev = sqrt(std_dev / ITER - avg_time*avg_time);    
printf("pipeline demonstrator, step 2b: \n"
"\nsum is %g\ntime is :%g (min %g, std_dev %g, all %g)\n",
sum, avg_time, min_time, std_dev, tend_all);
double max_GB_per_sec          = (double)_MEM_CLOCK / 1000 * (_MEM_WIDTH / 8);
double transfer_rate_in_GB_sec = (double)N*2*sizeof(double) / (1024*1024*1024) / avg_time;  
printf("transfer rate was %6.3g GB/sec (%7.4g%% of theoretical max that is %5.2g GB/sec)\n",
transfer_rate_in_GB_sec, transfer_rate_in_GB_sec / max_GB_per_sec * 100, max_GB_per_sec);
#ifdef USE_PAPI
printf( "\tIPC: %4.2g\n"
"\t[ time: %6.4gsec  -  ins: %9.6g ]\n",
ipc,
proctime / ITER,
(double)ins / ITER );
#endif
memset ( array2, 0, sizeof(double)*N );
memset ( array1, 0, sizeof(double)*N );
free( array2 );
free( array1 );
return 0;
}
