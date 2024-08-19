#include <math.h>
#include <omp.h>
void a(int *x, int *y, int n) {
double factor = 1;
for (int i = 0; i < n; i++) {
x[i] = factor * y[i];
factor = factor / 2;
}
}
void a_sol(int *x, int *y, int n) {
double factor = 1;
#pragma omp parallel for schedule(guided)
for (int i = 0; i < n; i++) {
x[i] = (factor / pow(2, i)) * y[i];
}
}
void b(int *x, int *y, int *z, int n) {
for (int i = 1; i < n; i++) {
x[i] = (x[i] + y[i - 1]) / 2;
y[i] = y[i] + z[i] * 3;
}
}
void b_sol(int *x, int *y, int *z, int n) {
#pragma omp parallel
{
#pragma omp for
for (int i = 1; i < n; i++) {
y[i] = y[i] + z[i] * 3;
}
#pragma omp for
for (int i = 1; i < n; i++) {
x[i] = (x[i] + y[i - 1]) / 2;
}
};
}
void c(int *x, int *y, int n, int twice) {
x[0] = x[0] + 5 * y[0];
for (int i = 1; i < n; i++) {
x[i] = x[i] + 5 * y[i];
if (twice) {
x[i - 1] = 2 * x[i - 1];
}
}
}
void c_sol(int *x, int *y, int n, int twice) {
x[0] = x[0] + 5 * y[0];
#pragma omp parallel
{
#pragma omp for
for (int i = 1; i < n; i++) {
x[i] = x[i] + 5 * y[i];
}
if (twice) {
#pragma omp for
for (int i = 1; i < n; i++) {
x[i - 1] = 2 * x[i - 1];
}
}
}
