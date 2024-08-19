#include <omp.h>
#include <unistd.h>
#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <papi.h>
#include <math.h>
#define Niter (10)
#define N (4096)
#define threshold (0.00000001)
#define PAPI_ERROR_CHECK(X) \
if((X)!=PAPI_OK) \
{fprintf(stderr,"PAPI Error \n"); exit(-1);}
void papi_print_helper(const char* msg, long long *values);
void compare(int n, double wref[n], double w[n]);
double m[N][N], x[N], y[N], z[N], yy[N], zz[N];
int main(int argc, char *argv[])
{
int nThreads;
double rtclock(void);
double clkbegin, clkend;
double t1, t2;
int i, j, t;
if ( argc != 2 )
{
printf("Number of threads not specified\n");
exit(-1);
}
nThreads = atoi(argv[1]);
if ( nThreads <= 0 )
{
printf("Num threads <= 0\n");
exit(-1);
}
printf("Num threads = %d\n", nThreads );
omp_set_num_threads(nThreads);
printf("Matrix Size = %d\n", N);
for(i = 0; i < N; i++)
{
x[i] = 1.0 / (i + 1);
y[i] = z[i] = 0.0;
for(j = 0; j < N; j++)
{
m[i][j] = drand48();
}
}
#ifdef ENABLE_PAPI
int event_set = PAPI_NULL;
PAPI_library_init(PAPI_VER_CURRENT);
PAPI_ERROR_CHECK(PAPI_create_eventset(&event_set));
PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_DP_OPS));
PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_VEC_DP));
PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_L3_TCM));
PAPI_ERROR_CHECK(PAPI_add_event(event_set, PAPI_RES_STL));
long_long papi_values[4];
PAPI_ERROR_CHECK(PAPI_start(event_set));
#endif
t1 = rtclock();
for(t = 0; t < Niter; t++)
{
for(i = 0; i < N; i++)
for(j = 0; j < N; j++)
{
y[j] = y[j] + m[i][j] * x[i];
z[j] = z[j] + m[j][i] * x[i];
}
for (i = 0; i < N; i++)
{
x[i] += 1e-8;
}
}
#ifdef ENABLE_PAPI
PAPI_ERROR_CHECK(PAPI_stop(event_set, papi_values));
papi_print_helper("Base Version",papi_values);
#endif
t2 = rtclock();
printf("Base version: %.2f GFLOPs; Time = %.2f\n", 4.0e-9 * Niter * N * N / (t2 - t1), t2 - t1);
for(i = 0; i < N; i++)
{
x[i] = 1.0 / (i + 1);
yy[i] = y[i];
zz[i] = z[i];
y[i] = z[i] = 0.0;
}
omp_set_num_threads(nThreads);
#ifdef ENABLE_PAPI
PAPI_ERROR_CHECK(PAPI_start(event_set));
#endif
t1 = rtclock();
#pragma omp parallel private(t,i,j)
{
int num_threads = omp_get_num_threads();
int thread_id = omp_get_thread_num();
double block_size_dbl = N/(double)num_threads;
int block_size = ceil(block_size_dbl);
for(t = 0; t < Niter; t++)
{
for(i = 0; i < N; i++)
for(j = thread_id * block_size; j < (thread_id +1) * block_size && j< N; j++)
{
y[j] = y[j] + m[i][j] * x[i];
}
for(j = thread_id * block_size; j < (thread_id +1) * block_size && j< N; j++)
for(i = 0; i < N; i++)
{
z[j] = z[j] + m[j][i] * x[i];
}
#pragma omp barrier
for(i = thread_id * block_size; i < (thread_id +1) * block_size && i< N; i++)
{
x[i] += 1e-8;
}
#pragma omp barrier
}
}
#ifdef ENABLE_PAPI
PAPI_ERROR_CHECK(PAPI_stop(event_set, papi_values));
papi_print_helper("Optimized Version",papi_values);
#endif
t2 = rtclock();
printf("Optimized version: %.2f GFLOPs; Time = %.2f\n", 4.0e-9 * Niter * N * N / (t2 - t1), t2 - t1);
printf("Comparing y: ");
compare(N, yy, y);
printf("Comparing z: ");
compare(N, zz, z);
}
double rtclock(void)
{
struct timezone Tzp;
struct timeval Tp;
int stat;
stat = gettimeofday (&Tp, &Tzp);
if (stat != 0)
{
printf("Error return from gettimeofday: %d", stat);
}
return(Tp.tv_sec + Tp.tv_usec * 1.0e-6);
}
void compare(int n, double wref[n], double w[n])
{
double maxdiff, this_diff;
int numdiffs;
int i;
numdiffs = 0;
maxdiff = 0;
for (i = 0; i < n; i++)
{
this_diff = wref[i] - w[i];
if (this_diff < 0)
{
this_diff = -1.0 * this_diff;
}
if (this_diff > threshold)
{
numdiffs++;
if (this_diff > maxdiff)
{
maxdiff = this_diff;
}
}
}
if (numdiffs > 0)
printf("%d Diffs found over threshold %f; Max Diff = %f\n",
numdiffs, threshold, maxdiff);
else
{
printf("Passed Correctness Check\n");
}
}
void papi_print_helper(const char* msg, long long *values)
{
printf("\n=====================PAPI COUNTERS==========================\n\n");
printf("(%s): No of DP operations : %.2f G\n",          msg, values[0]*1e-9);
printf("(%s): No of DP vector instructions : %.2f M\n", msg, values[1]*1e-6);
printf("(%s): L3 cache misses : %.2f M\n",              msg, values[2]*1e-6);
printf("(%s): Resource Stall Cycles: %.2f M\n",         msg, values[3]*1e-6);
printf("=================PAPI COUNTERS END==========================\n\n");
}
