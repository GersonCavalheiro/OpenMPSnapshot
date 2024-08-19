#define MATMUL_1D_OPEN_MP
#ifdef MATMUL_1D_OPEN_MP



#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>


void init_seq(double *a, const unsigned n_rows_a, const unsigned n_cols_a) {
int i;

#pragma omp parallel for default(none) private(i) shared(a, n_rows_a, n_cols_a) schedule(static)
for (i = 0; i < n_rows_a; i++) {
for (size_t j = 0; j < n_cols_a; j++) {
a[i*n_cols_a + j] = i*n_cols_a + j;
}
}
}


void init_rand(double *a, const unsigned n_rows_a, const unsigned n_cols_a) {
int i;


#pragma omp parallel for default(none) private(i) shared(a, n_rows_a, n_cols_a) schedule(dynamic)
for (i = 0; i < n_rows_a; i++) {
for (size_t j = 0; j < n_cols_a; j++) {
a[i*n_cols_a + j] = rand() / (double)RAND_MAX;
}
}
}


double *transpose(const double *m, const unsigned n_rows_m, const unsigned n_cols_m, double *t) {
int i, j;

#pragma omp parallel for default(none) private(i, j) shared(m, n_rows_m, n_cols_m, t) schedule(static)
for (i = 0; i < n_rows_m; i++) {
for (j = 0; j < n_cols_m; j++) {
t[j*n_rows_m + i] = m[i*n_cols_m + j];
}
}

return t;
}


double *dot_simple(const double *a, const unsigned n_rows_a, const unsigned n_cols_a, \
const double *b, const unsigned n_rows_b, const unsigned n_cols_b) {

if (n_cols_a != n_rows_b) {
printf("#columns A must be equal to #rows B!\n");
system("pause");
exit(-2);
}

double *c = malloc(n_rows_a * n_cols_b * sizeof(*c));
if (c == NULL) {
printf("Couldn't allocate memory!\n");
system("pause");
exit(-1);
}

int i, j, k;

#pragma omp parallel for default(none) private(i, j, k) shared(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b, c) schedule(static)
for (i = 0; i < n_rows_a; i++) {
for (k = 0; k < n_cols_b; k++) {
double sum = 0.0;
for (j = 0; j < n_cols_a; j++) {
sum += a[i*n_cols_a + j] * b[j*n_cols_b + k];
}
c[i*n_cols_b + k] = sum;
}
}

return c;
}


double *dot(const double *a, const unsigned n_rows_a, const unsigned n_cols_a, \
const double *b, const unsigned n_rows_b, const unsigned n_cols_b) {

int i, j, k;

if (n_cols_a != n_rows_b) {
printf("#columns A must be equal to #rows B!\n");
system("pause");
exit(-2);
}

double *bt = malloc(n_rows_b * n_cols_b * sizeof(*b));

double *c = malloc(n_rows_a * n_cols_b * sizeof(*c));

if ((c == NULL) || (bt == NULL)) {
printf("Couldn't allocate memory!\n");
system("pause");
exit(-1);
}

bt = transpose(b, n_rows_b, n_cols_b, bt);

#pragma omp parallel for default(none) private(i, j, k) shared(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b, c, bt) schedule(static)
for (i = 0; i < n_rows_a; i++) {
for (k = 0; k < n_cols_b; k++) {
double sum = 0.0;
for (j = 0; j < n_cols_a; j++) {
sum += a[i*n_cols_a + j] * bt[k*n_rows_b + j];
}
c[i*n_cols_b + k] = sum;
}
}

free(bt);

return c;
}


void print(const double *a, const unsigned n_rows_a, const unsigned n_cols_a) {
for (size_t i = 0; i < n_rows_a; i++) {
for (size_t j = 0; j < n_cols_a; j++) {
printf("%8.3f ", a[i*n_cols_a + j]);
}
printf("\n");
}
printf("\n");
}

int main(int argc, char *argv[]) {

time_t t;
srand((unsigned)time(&t));
srand(0);

omp_set_num_threads(4);
printf("omp_get_num_procs %i\n", omp_get_num_procs());
printf("omp_get_max_threads %i\n", omp_get_max_threads());
puts("");


double t0, t1;

const unsigned scale = 400;
const unsigned n_rows_a = 4 * scale;
const unsigned n_cols_a = 3 * scale;
const unsigned n_rows_b = 3 * scale;
const unsigned n_cols_b = 2 * scale;

double *a = malloc(n_rows_a * n_cols_a * sizeof(*a));
double *b = malloc(n_rows_b * n_cols_b * sizeof(*b));
double *c = NULL;
double *d = NULL;

if (!a || !b) {
printf("Couldn't allocate memory!\n");
system("pause");
exit(-1);
}

init_rand(a, n_rows_a, n_cols_a);
init_rand(b, n_rows_b, n_cols_b);

init_seq(a, n_rows_a, n_cols_a);
init_seq(b, n_rows_b, n_cols_b);

t0 = omp_get_wtime();
c = dot_simple(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b);
t1 = omp_get_wtime();
printf("OpenMP Dot Simple: Elapsed time %.3f s\n", t1 - t0);

t0 = omp_get_wtime();
d = dot(a, n_rows_a, n_cols_a, b, n_rows_b, n_cols_b);
t1 = omp_get_wtime();
printf("OpenMP Dot: Elapsed time %.3f s\n", t1 - t0);

if (scale == 1) {
printf("Matrix A:\n");
print(a, n_rows_a, n_cols_a);
printf("Matrix B:\n");
print(b, n_rows_b, n_cols_b);
printf("Matrix C:\n");
print(c, n_rows_a, n_cols_b);
printf("Matrix D:\n");
print(d, n_rows_a, n_cols_b);
}

free(a);
free(b);
free(c);
free(d);

system("pause");
return(0);
}

#endif 
