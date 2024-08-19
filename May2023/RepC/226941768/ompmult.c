#include "ompmult.h"
#include "dot_product.h"
#include "omp.h"
matrix *omp_mult_matrix_by_matrix(matrix *A, matrix *B) {
unsigned long i, j;
matrix *C = create_matrix(A->lin, B->lin);
#pragma omp parallel for private (i, j)
for (i = 0; i < A->lin; i++) {
for (j = 0; j < B->lin; j++) {
C->val[i][j] = dot_product(A->val[i], B->val[j], B->col);
}
}
return C;
}