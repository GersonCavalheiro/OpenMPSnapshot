#include "NoFLA_Apply_td_BlockH_var1_hier.h"
#include "qrca_utils.h"
#include "qrca_kernels.h"
#include <stdio.h>
#include <stdlib.h>
int NoFLA_Apply_td_BlockH_var1_hier( int m, int n, int k,
double * buff_Ul, int ldim_Ul,
double * buff_S,  int ldim_S,
double * buff_W,  int ldim_W,
double * buff_C1, int ldim_C1,
double * buff_C2, int ldim_C2 ) 
{
double   d_one = 1.0, d_mone = -1.0, d_zero = 0.0;
if( ( k == 0 )||( n == 0 ) ) {
return 0;
}
int m1 = m / 2; 
int m2 = m - m1;
int nn = n*n;
int kn = k*n;
double *U1=buff_Ul;
double *U2=buff_Ul+m1;
double *C1=buff_C1;
double *C3=buff_C2;
double *C4=C3+m1;
double *F=(double*)malloc(sizeof(double)*kn*3);
double *work1=F+kn;
double *work2=work1+kn;
NoFLA_Copy(k, n, buff_C1, ldim_C1, F, n);
dgemm_("Transpose", "No Transpose", &k, &n, &m1, 
&d_one, U1, &ldim_Ul,
C3, &ldim_C2,
&d_one, F, &k);	
dgemm_("Transpose", "No Transpose", &k, &n, &m2, 
&d_one, U2, &ldim_Ul,
C4, &ldim_C2,
&d_one, F, &n);	
split_dlarfb_hier_task(m1, n, C3, ldim_C2, U1, ldim_Ul, k, buff_S, ldim_S, k, F, work1);
split_dlarfb_hier_task(m2, n, C4, ldim_C2, U2, ldim_Ul, k, buff_S, ldim_S, k, F, work2);
dgemm_("Transpose", "No Transpose", &k, &n, &k, 
&d_mone, buff_S, &ldim_S,
F, &k,
&d_one, C1, &ldim_C1);	
#pragma omp taskwait
free(F);
return 0;
}
