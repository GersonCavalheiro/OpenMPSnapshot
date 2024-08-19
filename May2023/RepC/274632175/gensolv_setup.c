#include "gensolv_setup.h"
#include <stdlib.h>
#include <time.h>
#include "fptype.h"
#include "fpblas.h"
#include "genmat.h"
#include "matfprint.h"
int gensolv_setup(int check, int m, int n, int b, fp_t **A, fp_t **B, fp_t **Aorig, fp_t **Borig)
{
int m2 = m * m;
fp_t *lA = *A = malloc(m2 * sizeof(fp_t));
#pragma omp register ([m2]lA)
GENMAT_IP(lA, m, m, 100);
fp_t *lA0 = *Aorig = malloc(m2 * sizeof(fp_t));
int i;
for ( i=0; i<m2; ++i)
lA0[i] = lA[i];
int mn = m * n;
fp_t *lB = *B = GENMAT(m, n, m);
#pragma omp register ([m*n]lB)
fp_t *lB0 = *Borig = malloc(mn * sizeof(fp_t));
for(i=0; i<mn; ++i)
lB0[i] = lB[i];
return 0;
}
void gensolv_shutdown(fp_t *A, fp_t *B, fp_t *Aorig, fp_t *Borig) 
{
free(A);
free(B);
free(Aorig);
free(Borig);
}
