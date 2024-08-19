#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 100000
#define REPETITIONS 100000
int main(int argc, void **argv) {
int n;
if (argc == 2) {
n = atoi(argv[1]);
} else {
n = N;
fprintf(stderr, "WARN: No args, using default values \n");
}
printf("N=%d\n", n);
float *a = (float *)malloc(sizeof(float) * n + 16);
float *b = (float *)malloc(sizeof(float) * n + 16);
float *c = (float *)malloc(sizeof(float) * n + 16);
if (a == NULL || b == NULL || c == NULL) {
puts("MALLOC ERROR");
return EXIT_SUCCESS;
}
for (int i = 0; i < n; i++) {
a[i] = i * 0.001;
b[i] = i * 0.002;
c[i] = i * 0.003;
}
double start = omp_get_wtime();
for (int run = 0; run < REPETITIONS; ++run) {
#pragma omp simd aligned(a, b, c : 32)
for (int i = 0; i < N; ++i) {
a[i] += b[i] * c[i];
}
}
double end = omp_get_wtime();
printf("Executiontime: %lf\n", end - start);
for (int i = 0; i < 10 && i < N && i < REPETITIONS; ++i) {
printf("%f ", a[i]);
}
free(a);
free(b);
free(c);
return EXIT_SUCCESS;
}