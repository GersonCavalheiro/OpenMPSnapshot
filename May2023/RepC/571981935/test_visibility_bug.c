#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main() {
size_t n = 10000000;
double *x = malloc(n * sizeof *x);
double *y = malloc(n * sizeof *y);
double s = 0;
double reference_solution = 0;
double tmp = 0;
#pragma omp parallel for
for (int i = 0; i < n; ++i) {
x[i] = 1.23 * i;
y[i] = 4.56 * i;
}
reference_solution = 1.23 * 4.56 * (n-1) * n * (2*n-1) / 6;
#pragma omp parallel
{
#pragma omp for
for (int i = 0; i < n; i++)
tmp += x[i] * y[i];
#pragma omp atomic update
s += tmp;
}
printf("Expected result: %.5e\nComputed result: %.5e\n",
reference_solution, s);
free(x);
free(y);
}
