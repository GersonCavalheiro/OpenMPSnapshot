#include "sedm.h"
#include <omp.h>
#include <gsl/gsl_cblas.h>
#include "matrix.h"
#include "def.h"
void sedm_comp(const elem_t *X, size_t n, const elem_t *Y, size_t m, size_t d, elem_t *distance_matrix) {
int threadnum = omp_get_max_threads();
omp_set_num_threads(threadnum);
elem_t *X_sqrd = (elem_t*) malloc(n * sizeof(elem_t));
elem_t *Y_sqrd = (elem_t*) malloc(m * sizeof(elem_t));
#pragma omp parallel for
for (size_t i = 0; i < n; i++) {
VEC_T x_sqrd = VEC_ZERO();
for (size_t k = 0; k < d - d % VEC_SIZE; k += VEC_SIZE) {
VEC_T x = VEC_LOAD(MATRIX_ROW(X, i, n, d) + k);
x_sqrd = VEC_FMADD(x, x, x_sqrd);
}
elem_t sum = VEC_SUM(x_sqrd);
for (size_t k = d - d % VEC_SIZE; k < d; k++) {
elem_t x = MATRIX_ELEM(X, i, k, n, d);
sum += x * x;
}
X_sqrd[i] = sum;
}
#pragma omp parallel for
for (size_t j = 0; j < m; j++) {
VEC_T y_sqrd = VEC_ZERO();
for (size_t k = 0; k < d - d % VEC_SIZE; k += VEC_SIZE) {
VEC_T y = VEC_LOAD(MATRIX_ROW(Y, j, m, d) + k);
y_sqrd = VEC_FMADD(y, y, y_sqrd);
}
elem_t sum = VEC_SUM(y_sqrd);
for (size_t k = d - d % VEC_SIZE; k < d; k++) {
elem_t y = MATRIX_ELEM(Y, j, k, m, d);
sum += y * y;
}
Y_sqrd[j] = sum;
}
elem_t *XY = (elem_t*) malloc(n * m * sizeof(elem_t));
#pragma omp parallel for
for(size_t t = 0 ; t < threadnum ; t++) {
size_t i_begin = t * n / threadnum;
size_t i_end = min((t + 1) * n / threadnum, n);
const elem_t *Xi = MATRIX_ROW(X, i_begin, n, d);
elem_t *XYi = MATRIX_ROW(XY, i_begin, n, m);
cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 
i_end - i_begin, m, d, 
1.0, Xi, d, Y, d, 
0.0, XYi, m);
}
#pragma omp parallel for
for (size_t i = 0; i < n; i++) {
for (size_t j = 0; j < m; j++) {
MATRIX_ELEM(distance_matrix, i, j, n, m) = X_sqrd[i] - 2 * MATRIX_ELEM(XY, i, j, n, m) + Y_sqrd[j];
}
}
free(X_sqrd);
free(Y_sqrd);
free(XY);
}
void sedm_simp(const elem_t *X, size_t n, const elem_t *Y, size_t m, size_t d, elem_t *distance_matrix) {
int threadnum = omp_get_max_threads();
omp_set_num_threads(threadnum);
#pragma omp parallel for
for (size_t i = 0; i < n; i++) {
for (size_t j = 0; j < m; j++) {
VEC_T v_sum = VEC_ZERO();
for (size_t k = 0; k < d - d % VEC_SIZE; k += VEC_SIZE) {
VEC_T x = VEC_LOAD(MATRIX_ROW(X, i, n, d) + k);
VEC_T y = VEC_LOAD(MATRIX_ROW(Y, j, m, d) + k);
VEC_T diff = VEC_SUB(x, y);
v_sum = VEC_FMADD(diff, diff, v_sum);
}
elem_t sum = VEC_SUM(v_sum);
for(size_t k = d - d % VEC_SIZE; k < d; k++) {
elem_t x = MATRIX_ELEM(X, i, k, n, d);
elem_t y = MATRIX_ELEM(Y, j, k, m, d);
elem_t diff = x - y;
sum += diff * diff;
}
MATRIX_ELEM(distance_matrix, i, j, n, m) = sum;
}
}
}
