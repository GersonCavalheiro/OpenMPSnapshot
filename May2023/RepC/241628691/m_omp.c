#include "m_omp.h"
#include <complex.h>
#include <omp.h>
#include "matrix.h"
void multiply_omp(const matrix_struct *a, const matrix_struct *b, matrix_struct *result, int num_threads) {
int i, j, k, ii;
unsigned int row_size = a->rows;
double complex *conjTN = (double complex *) emalloc(num_threads*row_size* sizeof(double complex));
double complex *conjT = NULL;
omp_set_num_threads(num_threads);
#pragma omp parallel for private(i, j, k, ii, conjT)
for (i = 0; i < result->rows; ++i)
for (k = 0; k < a->rows; ++k) {
conjT = conjTN + row_size*omp_get_thread_num();
for (ii = 0; ii < result->rows; ++ii)
conjT[ii] = conj(a->data[k][ii]);
for (j = 0; j < result->cols; ++j) {
result->data[i][j] += conjT[i] * b->data[k][j];
}
}
free(conjTN);
}