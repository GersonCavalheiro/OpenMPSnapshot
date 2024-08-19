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
#if defined(_OPENMP)
#define CPU_TIME (clock_gettime( CLOCK_REALTIME, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#define CPU_TIME_th (clock_gettime( CLOCK_THREAD_CPUTIME_ID, &myts ), (double)myts.tv_sec +     \
(double)myts.tv_nsec * 1e-9)
#else
#define CPU_TIME (clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &ts ), (double)ts.tv_sec + \
(double)ts.tv_nsec * 1e-9)
#endif
#if defined(TASKS_GRANULARITY )
#define ROUND_N_TO_GRANULARITY {N += (N%TASKS_GRANULARITY);   printf("tasks will be created with granularity: %d\n", TASKS_GRANULARITY);}
#define CREATE_TASKS for ( int i = 0; i < N; i+= TASKS_GRANULARITY )
#define TASK_FOR for ( int JJ = i; JJ < i+TASKS_GRANULARITY; JJ++ )
#define TASKS_SIZE TASKS_GRANULARITY
#else
#define ROUND_N_TO_GRANULARITY 
#define CREATE_TASKS for ( int i = 0; i < N; i++ )
#define TASK_FOR
#define JJ i
#define TASKS_SIZE 1
#endif
#if defined(RANDOMLY_DECREASING)
#define DECREASING_WORK( I ) workload - ((I) + rand_r(&seeds[me]) % (10+((I)/10)))
#else
#define DECREASING_WORK( I ) workload - (I)
#endif
#define REPETITIONS 10
#define NSTRATEGIES 2
#define FOR         0
#define TASKS       1
char *STRATEGIES_NAMES[] = {"FORloop", "TASKS"};
#define NTIMINGS    1
#define RND_WORK    0
char *TIMINGS_NAMES[] = {"RANDOM work"};
double heavy_work( int N );
int main( int argc, char **argv )
{
int nthreads;
int N = 10000;
int workload = 40000;
double wtstart, wtend;
struct  timespec ts;
if ( argc > 1 )
{
N = atoi( *(argv+1) );
if ( argc > 2 )
workload = atoi(*(argv+2));
}
#pragma omp parallel
#pragma omp master
nthreads = omp_get_num_threads();
printf("using %d threads with N = %d\n\n", nthreads, N);
double timings[NTIMINGS][NSTRATEGIES][nthreads];
double wtimings[NTIMINGS][NSTRATEGIES] = {0.0};
double min_timings[NTIMINGS][NSTRATEGIES] = {0.0};
double max_timings[NTIMINGS][NSTRATEGIES] = {0.0};
#if defined(DEBUG)
unsigned int howmanytasks[nthreads];
#endif
memset( timings, 0, NTIMINGS*NSTRATEGIES*nthreads*sizeof(double));
#if defined(DEBUG)
memset( howmanytasks, 0, nthreads*sizeof(int));
#endif
ROUND_N_TO_GRANULARITY ;
for ( int R = 0; R < REPETITIONS; R++ )
{
printf("shot %d/%d.. ", R+1, REPETITIONS);
fflush(stdout);
wtstart = CPU_TIME;
#pragma omp parallel shared(N, workload)
{
struct  timespec myts;
int myid   = omp_get_thread_num();
int status = myid;
unsigned int half_workload = workload/2;
srand( myid*123 + time(NULL) );
double tstart = CPU_TIME_th;
#pragma omp for schedule(dynamic, TASKS_SIZE)
for( int i = 0; i < N; i++ )
{
unsigned int work = 10 + rand_r(&status) % workload;
if ( work > half_workload )
heavy_work( work );
}
double tend = CPU_TIME_th;
timings[RND_WORK][FOR][myid] += tend - tstart;
}
wtend = CPU_TIME;
wtimings[RND_WORK][FOR] += wtend - wtstart;
unsigned int seeds[nthreads];
wtstart = CPU_TIME;
#pragma omp parallel shared(N, workload)
{
struct  timespec myts;
int myid   = omp_get_thread_num();
srand( myid*123 + time(NULL) );
double tstart = CPU_TIME_th;
#pragma omp single nowait
{
unsigned int half_workload = workload/2;
CREATE_TASKS
#pragma omp task
{
int me = omp_get_thread_num();
TASK_FOR
{
unsigned int work = 10 + rand_r(&seeds[me]) % workload;
if ( work > half_workload )
heavy_work( work );
}
}
}
#pragma omp barrier
double tend = CPU_TIME_th;
timings[RND_WORK][TASKS][myid] += tend - tstart;
}
wtend = CPU_TIME;
wtimings[RND_WORK][TASKS] += wtend - wtstart;
}    
double INV_REP = 1.0 / REPETITIONS;
for ( int k = 0; k < NTIMINGS; k++ )
{
printf("\ntimings %s:\n", TIMINGS_NAMES[k] );
double std_dev = 0;
for ( int j = 0; j < NSTRATEGIES; j++ )
{	  
min_timings[k][j] = timings[k][j][0];
max_timings[k][j] = timings[k][j][0];
std_dev = timings[k][j][0]*timings[k][j][0];
for( int i = 1; i < nthreads; i++)
{
timings[k][j][0] += timings[k][j][i];
std_dev          += timings[k][j][i] * timings[k][j][i];
min_timings[k][j] = (min_timings[k][j] < timings[k][j][i]) ? min_timings[k][j] : timings[k][j][i];
max_timings[k][j] = (max_timings[k][j] > timings[k][j][i]) ? max_timings[k][j] : timings[k][j][i];
}
timings[k][j][0] /= nthreads;
std_dev = sqrt( std_dev/(nthreads-1) - nthreads/(nthreads-1)*timings[k][j][0]*timings[k][j][0] );
printf("\t%16s :  w-clock %9.7g, avg %9.7g +- %9.7g, min: %9.7g, max: %9.7g\n",
STRATEGIES_NAMES[j],
wtimings[k][j]*INV_REP, timings[k][j][0]*INV_REP, std_dev*INV_REP, min_timings[k][j]*INV_REP, max_timings[k][j]*INV_REP );
}
}
#if defined(DEBUG)
for ( int t = 0; t < nthreads; t++ )
printf("thread %d has processed %u tasks\n", t, howmanytasks[t] );
#endif
return 0;
}
double heavy_work( int N )
{
double guess = 3.141572 / 5 * N;
guess = ( guess > 200 ? 111 : guess);
for( int i = 0; i < N; i++ )
{
guess = exp( guess );
guess = sin( guess );
}
return guess;
}
