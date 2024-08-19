#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "fptype.h"
#include "fpmatr.h"
#include "matmul_config.h"
#include "matmul_setup.h"
#include "matmul_check.h"
#include "mm_main.h"
#include "csrmmb_main.h"
#include "csrmm_main.h"
#include "matfprint.h"
#include "selfsched.h"
#include "file_log.h"
int m;
int n;
int k;
int b;
int d;
int c;
int lda;
int ldb;
int ldc;
fp_t alpha;
fp_t beta;
unsigned int l;
int reps;
int check;
ompssblas_t transa;
ompssblas_t transb;
void *A;
fp_t *B;
fp_t *C;
char *fname;
selfsched_t *sched;
int schedc;
log_f logf_gemm;
int main(int argc, char* argv[]) 
{
if ( matmul_config(argc, argv) ) {
return 1;
}
if ( matmul_setup(fname, m, n, k, b, d, c, &A, &B, &C) ) {
return 2;
}
logf_init(&logf_gemm, reps, "matmul");
int r;
for ( r = 0; r < reps; ++r ) {
logf_record_stime(&logf_gemm);
#if USE_GEMM
MM_MAIN(b, d, c, m, n, k, transa, transb, alpha, A, lda, B, ldb, beta, C, ldc);
#elif USE_CSRMMB
hbmat_t *Ahbh = (hbmat_t*) A;
hbmat_t *Ahb = Ahbh->orig;
m = Ahb->m;
k = Ahb->n;
CSRMMB(b, d, c, n, FP_ONE, A, B, k, FP_NOUGHT, C, m);
#elif USE_CSRMM
hbmat_t *Ahbh = (hbmat_t*) A;
hbmat_t *Ahb = Ahbh->orig;
m = Ahb->m;
k = Ahb->n;
sched_alldone(schedc, sched, r);
CSRMM(b, d, c, n, FP_ONE, A, B, k, FP_NOUGHT, C, m, sched);
#endif
#pragma omp taskwait
logf_record_etime(&logf_gemm);
}
logf_dump(&logf_gemm);
#if USE_GEMM
gemm_check(check, b, d, c, m, n, k, transa, transb, alpha, A, lda, B, ldb, beta, C, ldc);
#else
matmul_check(check, b, d, c, m, n, k, 0, 0, A, B, k, C, m);
#endif
matmul_shutdown(m, n, k, b, d, c, A, B, C);
return 0;
}
