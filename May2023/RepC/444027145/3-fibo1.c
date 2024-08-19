#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
void fibo_serial(int n) {
int i;
long long f1, f2, t;
printf("serial, the first %d Fibonacci numbers:\n", n);
f1 = f2 = 1;
printf("%lld %lld ", f1, f2);
for (i = 2; i < n; i++) {
t = f2;
f2 = f1 + f2;
f1 = t;
printf("%lld ", f2);
}
printf("\n");
}
void fibo_serial_arr(int n) {
int i;
long long f1, f2, t;
long long *fibo = malloc(n * sizeof(long long));
printf("serial, the first %d Fibonacci numbers:\n", n);
fibo[0] = fibo[1] = 1;
for (i = 2; i < n; i++)
fibo[i] = fibo[i - 1] + fibo[i - 2];
for (i = 0; i < n; i++)
printf("%lld ", fibo[i]);
printf("\n");
free(fibo);
}
int main(int argc, char *argv[])
{
int thread_count, n, i;
thread_count = 4;
n = 15;
if (argc > 1) {
thread_count = atoi(argv[1]);
}
if (argc > 2) {
n = atoi(argv[2]);
}
long long *fibo = malloc(n * sizeof(long long));
fibo[0] = fibo[1] = 1;
#pragma omp parallel for num_threads(thread_count)
for (i = 2; i < n; i++)
fibo[i] = fibo[i - 1] + fibo[i - 2];
printf("The first %d Fibonacci numbers:\n", n);
for (i = 0; i < n; i++)
printf("%lld ", fibo[i]);
printf("\n");
free(fibo);
fibo_serial(n);
fibo_serial_arr(n);
return 0;
}
