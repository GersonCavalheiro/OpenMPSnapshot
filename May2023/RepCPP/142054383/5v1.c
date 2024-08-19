#include <stdio.h>
#include <omp.h>

#define NUM_STEPS 10000
#define MAX_THREADS 500

double pi_uniprocess()
{
int i;
double x, step = 1.0/NUM_STEPS, sum = 0;

for(i=0; i<NUM_STEPS; i++)
{
x = (i+0.5)*step;
sum += 4.0/(1+x*x);
}

return sum * step;
}

double pi_multiprocess(int num_threads)
{
omp_set_num_threads(num_threads);
int i;
double x, step = 1.0/NUM_STEPS, sum = 0;

#pragma omp parallel private(i)
{
int thread_id = omp_get_thread_num();
double x, partial_sum = 0;

for(i=thread_id; i<NUM_STEPS; i+=num_threads)
{
x = (i+0.5)*step;
partial_sum += 4.0/(1+x*x);
}

#pragma omp atomic

sum += partial_sum;

}

return sum * step;
}

int main()
{
double start, end;

start = omp_get_wtime();
double pi = pi_uniprocess();
end = omp_get_wtime();

printf("Pi value from uniprocessor: %lf\tTime elapsed: %lfs\n\n", pi, end - start);
int i;

for(i=2; i<MAX_THREADS; i++)
{
start = omp_get_wtime();
pi = pi_multiprocess(i);
end = omp_get_wtime();

printf("Pi value from %d threads: %lf\tTime elapsed: %lfs\n", i, pi, end - start);
}
printf("\n");
}
