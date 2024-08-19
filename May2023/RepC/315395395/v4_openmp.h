#include <stdio.h>
#include <omp.h>
#ifndef matrixFunctions
#define matrixFunctions
#include "../include/matrixFunctions.h"
#endif
void triangleCountV4(uint32_t N, uint32_t *c, uint32_t *csr_row_ptr, uint32_t *csr_col, uint32_t num_threads) {
#pragma omp parallel for schedule(dynamic, 256)
for (int i = 0; i < N; i++) {
for (int j = csr_row_ptr[i]; j < csr_row_ptr[i + 1]; j++) {
multiplyRowCol(i, csr_col[j], c, csr_row_ptr, csr_col);
}
c[i] /= 2;
}
}