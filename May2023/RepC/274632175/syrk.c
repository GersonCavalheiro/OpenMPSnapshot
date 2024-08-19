#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>
#include "fptype.h"
#include "fpmatr.h"
#include "syrk_config.h"
#include "syrk_setup.h"
#include "syrk_check.h"
#include "syrk_main.h"
#include "file_log.h"
ompssblas_t uplo;
ompssblas_t trans;
fp_t alpha;
fp_t beta;
int n;
int k;
int b;
int lda;
int ldc;
int reps;
int check;
fp_t *A;
fp_t *C;
fp_t *Cc;
log_f logf_syrk;
int main(int argc, char* argv[]) 
{
if ( syrk_config(argc, argv) ) {
return 1;
}
if ( syrk_setup(n, k, b, &A, &C, &Cc) ) {
return 2;
}
logf_init(&logf_syrk, reps, "syrk");
int r;
for ( r = 0; r < reps; ++r ) {
logf_record_stime(&logf_syrk);
#ifdef USE_SYRK
SYRK_MAIN(uplo, trans, b, n, k, alpha, A, lda, beta, C, ldc);
#endif
#pragma omp taskwait
logf_record_etime(&logf_syrk);
}
fp_t ret = syrk_check(check, uplo, trans, b, n, k, alpha, A, lda, beta, C, ldc, Cc);
logf_dump(&logf_syrk);
if (check){
char tmp[32];
snprintf(tmp, 32, "relerr: %e\n", ret);
logf_append(&logf_syrk, tmp);
}
syrk_shutdown(n, k, b, A, C, Cc);
return 0;
}
