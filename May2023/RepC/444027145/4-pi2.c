#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
int main(int argc, char *argv[]) {
long n, i;
int thread_count;
double sum, factor;
thread_count = 4;
n = 1e9; 
if (argc > 1) {
thread_count = atoi(argv[1]);
}
if (argc > 2) {
n = atol(argv[2]);
}
double t0 = omp_get_wtime();
sum = 0.0;
#pragma omp parallel for num_threads(thread_count) default(none) shared(n) private(factor, i) reduction(+ : sum) 
for (i = 0; i < n; i++) {
factor = (i % 2 == 0) ? 1.0 : -1.0;
sum += factor / (2 * i + 1);
}
double pi_approx = 4.0 * sum;
double t1 = omp_get_wtime();
printf("With n = %ld terms and %d threads,\n", n, thread_count);
printf("   Our estimate of pi = %.14f\n", pi_approx);
printf("                 M_PI = %.14f\n", M_PI);
printf("time = %.3f sec\n", t1 - t0);
return 0;
}
