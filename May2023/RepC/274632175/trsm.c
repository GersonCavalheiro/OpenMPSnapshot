#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "trsm_config.h"
#include "trsm_setup.h"
#include "trsm_main.h"
#include "trsm_check.h"
#include "fptype.h"
#include "matfprint.h"
#include "file_log.h"
int m; 
int n; 
int b; 
int reps; 
int check; 
fp_t alpha;
int lda;
int ldb;
ompssblas_t uplo; 
ompssblas_t side;
ompssblas_t trans;
ompssblas_t diag;
fp_t *A;
fp_t *B;
fp_t *Bchk;
log_f logf_tri;
int main(int argc, char* argv[]) 
{
if ( trsm_config(argc, argv) ) {
return 1;
}
if ( trsm_setup(check, m, n, b, lda, ldb, &A, &B, &Bchk) ) {
fprintf(stderr, "err: allocating matrix\n");
return 2;
}
logf_init(&logf_tri, reps, "trsm");
int r;
for (r = 0; r < reps; r++) {
logf_record_stime(&logf_tri);
TRSM_MAIN(side, uplo, trans, diag, m, n, b, alpha, A, lda, B, ldb);
#pragma omp taskwait
logf_record_etime(&logf_tri);
}
fp_t ret = trsm_check(check, A, B, Bchk);
logf_dump(&logf_tri);
if (check) {
char tmp[32];
snprintf(tmp, 32, "relerr: %e\n", ret);
logf_append(&logf_tri, tmp);
}
trsm_shutdown(A, B, Bchk);
return 0;
}
