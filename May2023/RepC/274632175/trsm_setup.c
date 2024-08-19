#include "trsm_setup.h"
#include <stdlib.h>
#include <time.h>
#include "fptype.h"
#include "fpblas.h"
#include "genmat.h"
#include "densutil.h"
#include "matfprint.h"
extern ompssblas_t side;
extern ompssblas_t uplo;
extern ompssblas_t trans;
extern ompssblas_t diag;
extern fp_t alpha;
int trsm_setup(int check, int m, int n, int b, int lda, int ldb, fp_t **A, fp_t **B, fp_t **Bchk) 
{
fp_t *lA = *A = malloc(lda*m*sizeof(fp_t));
if ( lA == NULL )
return 1;
#pragma omp register ([lda*m]lA)
GENMAT_IP(lA, lda, m, 1);
fp_t *lB = *B = malloc(ldb * n * sizeof(fp_t));
if (lB == NULL)
return 2;
#pragma omp register ([ldb*n]lB)
GENMAT_IP(lB, ldb, n, 1);
fp_t *lBchk = *Bchk = malloc(ldb * n * sizeof(fp_t));
if (lBchk == NULL)
return 3;
int i;
for ( i = 0; i < ldb*n; ++i )
lBchk[i] = lB[i];
return 0;
}
void trsm_shutdown(fp_t *A, fp_t *B, fp_t *X) 
{
free(A);
free(B);
free(X);
}
