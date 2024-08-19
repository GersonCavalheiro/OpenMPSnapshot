#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
double f(int i);
int main(int argc, char *argv[])
{
double sum;         
long n;             
int thread_count;
thread_count = 4;
n = 15000;
if (argc > 1) {
thread_count = atoi(argv[1]);
}
if (argc > 2) {
n = atol(argv[2]);
}
double t0 = omp_get_wtime();
sum = 0.0;
#pragma omp parallel for num_threads(thread_count) reduction(+ : sum) schedule(guided)
for (int i = 0; i < n; i++) {
sum += f(i);
}
double t1 = omp_get_wtime();
printf("Result = %.14f\n", sum);
printf("time   = %.3lf sec\n", t1 - t0);
return 0;
}
double f(int i)
{
int start = i * (i + 1) / 2;
int finish = start + i;
double return_val = 0.0;
for (int j = start; j <= finish; j++) {
return_val += sin(j);
}
return return_val;
}
