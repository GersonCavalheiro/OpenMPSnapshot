#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
long long fib(int n, long long * arr) {
long long f1, f2;
if (n < 2) {
return n;
}
else if (arr[n]) return arr[n];
else {
#pragma omp task shared(f1)
f1 = fib(n-1, arr);
#pragma omp task shared(f2)
f2 = fib(n-2, arr);
#pragma omp taskwait
arr[n] = f1 + f2;
return arr[n];
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
long long* arr = calloc(n + 1, sizeof(long long));
arr[1] = 1;
double t0 = omp_get_wtime();
#pragma omp parallel shared(n, arr)
{
#pragma omp single
{
for (int i = 1; i <= n; i++)
printf("%lld ", fib(i, arr));
printf("\n");
}
}
double t1 = omp_get_wtime();
printf("time = %.3f sec\n", t1 - t0);
}
