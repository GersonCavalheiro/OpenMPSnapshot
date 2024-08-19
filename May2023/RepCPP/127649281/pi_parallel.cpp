
#include <stdlib.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>

#define NUM_THREADS 4

int main() {

static long num_steps = 100000;
int i = 0;
double pi;
double step_width = 1.0 / (double)num_steps;
int ACTUAL_NUM_THREADS;

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel
{

int ID = omp_get_thread_num();
int i;
double x = (double)ID;
double y;
double sum;
int nthrds;

ACTUAL_NUM_THREADS = omp_get_num_threads();

for (i = ID; i < num_steps; i += ACTUAL_NUM_THREADS) {

x = i * step_width;

y = 4.0 / (1.0 + x * x);
sum += step_width * y;

}

#pragma omp critical
pi += sum;

}

printf("The computed pi number is equal to: %lf \n", pi);

return 0;
}
