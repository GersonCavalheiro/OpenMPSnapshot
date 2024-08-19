#include "gensolv_main.h"
#include "fpblas.h"
#include "fplapack.h"
#include "fpmatr.h"
#include "bblas_trsm.h"
#include "ompss_lu.h"
#include "lu_kernels.h"
int GENSOLV_MAIN(int m, int n, int b, fp_t *A, fp_t *B) 
{
int *IPIV = malloc(n * sizeof(int));
#pragma omp register ([n]IPIV)
OMPSS_LU(m, m, b, A, m, IPIV);
int i;
for ( i=0; i<n; i+=b ) {
int dim = n-i > b ? b : n-i;
int *ip = &IPIV[i];
TASK_LASWP(b, n, B, m, 1, dim, ip, 1);
}
BBLAS_LTRSM(OMPSSBLAS_LOWERTRIANG, OMPSSBLAS_NTRANSP, OMPSSBLAS_DIAGUNIT, b, n, m, n, FP_ONE, A, m, B, m);
BBLAS_LTRSM(OMPSSBLAS_UPPERTRIANG, OMPSSBLAS_NTRANSP, OMPSSBLAS_NDIAGUNIT, b, n, m, n, FP_ONE, A, m, B, m);
return 0;
}
