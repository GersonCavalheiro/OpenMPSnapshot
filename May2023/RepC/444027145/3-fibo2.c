#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
long long fib(int n) {
long long f1, f2;
if (n < 2)
return n;
else if (n < 20) return fib(n-1) + fib(n-2);
else {
#pragma omp task shared(f1)
f1 = fib(n-1);
#pragma omp task shared(f2)
f2 = fib(n-2);
#pragma omp taskwait
return f1 + f2;
}
}
int main(int argc, char* argv[]) {
int thread_count, n;
thread_count = 4;
n = 15;
if (argc > 2) {
thread_count = atoi(argv[1]);
n = atoi(argv[2]);
}
omp_set_num_threads(thread_count);
double t0 = omp_get_wtime();
#pragma omp parallel shared(n)
{
#pragma omp single 
{
for (int i = 1; i <= n; i++)
printf("%lld ", fib(i));
printf("\n");
}
}
double t1 = omp_get_wtime();
printf("time = %.3f sec\n", t1 - t0);
}
