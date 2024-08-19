#include "gaussian_omp.h"
#include <stdlib.h>
void Gaussian_Elimination_omp(int n, double *A, double *b, double *y, int thread_count) {
int i, j, k;
#pragma omp parallel num_threads(thread_count) default(none) private(i, j, k) shared(n, A, b, y)
for (k = 0; k < n; k++) {
#pragma omp for
for (j = k + 1; j < n; j++) {
A[k * n + j] = A[k * n + j] / A[k * n + k];
}
y[k] = b[k] / A[k * n + k];
A[k * n + k] = 1;
#pragma omp for
for (i = k + 1; i < n; i++) {
for (j = k + 1; j < n; j++) {
A[i * n + j] = A[i * n + j] - A[i * n + k] * A[k * n + j];
}
b[i] = b[i] - A[i * n + k] * y[k];
A[i * n + k] = 0;
}
}
}
void Back_Substitution_omp(int n, double *U, double *x, double *y, int thread_count) {
int i, k;
for (k = n - 1; k > -1; k--) {
#pragma omp parallel for schedule(static) num_threads(thread_count) default(none) private(i) shared(n, U, x, y, k)
for (i = k - 1; i > -1; i--) {
y[i] -= x[k] * U[i * n + k];
}
}
}
void Gaussian_getX_omp(int n, double *A, double *b, double *y, double *x, int thread_count) {
Gaussian_Elimination_omp(n, A, b, y, thread_count);
Back_Substitution_omp(n, A, x, y, thread_count);
}