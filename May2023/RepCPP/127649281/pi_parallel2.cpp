#include <omp.h>
#include <stdio.h>

static long num_steps = 100000;
double step;
#define NUM_THREADS 2

int main() {
double pi;
double sum;
double x;
int i;
step = 1.0/(double)num_steps;

omp_set_num_threads(NUM_THREADS);

#pragma omp parallel for reduction(+:sum)
for(i = 0; i<num_steps; i += 1) {
x = (i + 0.5) * step;
sum += 4.0 / (1.0 + x*x);
}

printf("The computed pi number is equal to: %lf \n", sum);
return 0;
}
