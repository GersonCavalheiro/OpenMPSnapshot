#include <stdio.h>
#include <time.h>
#include <omp.h>
#define INTERVALS 10000000
double a[INTERVALS], b[INTERVALS];
int main(int argc, char **argv)
{
double time2;
double start_time = omp_get_wtime();
double *to = b;
double *from = a;
int    time_steps = 100;
omp_set_dynamic(0);
omp_set_num_threads(4);
long i = 0;
from[0] = 1.0;
from[INTERVALS - 1] = 0.0;
to[0] = from[0];
to[INTERVALS - 1] = from[INTERVALS - 1];
#pragma omp parallel for
for(long i = 1; i < INTERVALS; i++)
from[i] = 0.0;
printf("Number of intervals: %ld. Number of time steps: %d\n", INTERVALS, time_steps);
while(time_steps-- > 0)
{
#pragma omp parallel for shared(from, to)
for(long i = 1; i < (INTERVALS - 1); i++)
to[i] = from[i] + 0.1*(from[i - 1] - 2*from[i] + from[i + 1]);
{
double* tmp = from;
from = to;
to = tmp;
}
}
time2 = (omp_get_wtime() - start_time);
printf("Elapsed time (s) = %f\n", time2);
return 0;
}
