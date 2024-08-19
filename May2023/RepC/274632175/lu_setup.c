#include <stdlib.h>
#include "lu_setup.h"
void lu_setup(fp_t **Ap, fp_t **Aorigp, int **IPIVp, int m, int n, int check) 
{
*Ap = malloc( m * n * sizeof(fp_t) );
fp_t *A=*Ap;
#pragma omp register ([m*n]A)
*Aorigp = malloc( m * n * sizeof(fp_t));
fp_t *Aorig = *Aorigp;
*IPIVp = calloc( n , sizeof(int) );
int *IPIV = *IPIVp;
#pragma omp register ([n]IPIV)
srand48( ( long int ) 652918 );
int j;
for (j = 0; j < n; j++) {
int i;
for (i = 0; i < m ; i++) {
Aorig[j*m+i] = A[j*m+i] = drand48();
if ( i == j ) {
Aorig[j*m+i] += 100;
A[j*m+i] += 100;
}
}    
}
}
void lu_shutdown(fp_t *A, fp_t *Aorig, int *IPIV)
{
free(A);
free(Aorig);
free(IPIV);
}
