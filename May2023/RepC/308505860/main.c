#include <omp.h>
#include <stdio.h>
int main(int argc, char **argv) {
int m, n;
m = atoi(argv[1]);
n = atoi(argv[2]);
int i, j;
double k;
double *V1, *V2, *X;
V1 = (double *)malloc(m * n * sizeof(double));
V2 = (double *)malloc(n * sizeof(double));
X = (double *)malloc(m * sizeof(double));
for (i = 0; i < m; i++) {
for (j = 0; j < n; j++) {
V1[i * n + j] = 1.0;
}
}
for (i = 0; i < n; i++) {
V2[i] = 1.0;
}
for (i = 0; i < m; i++) {
X[i] = 0.0;
}
#pragma omp parallel private(i)
{
for (i = 0; i < m; i++) {
k = 0.0;
#pragma omp for reduction(+ : k)
for (j = 0; j < n; j++) {
k += V1[i * n + j] * V2[j];
}
X[i] = k;
#pragma omp barrier
}
}
for (i = 0; i < m; i++) {
printf("%lf\n", X[i]);
}
free(V1);
free(V2);
free(X);
return 0;
}
