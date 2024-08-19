#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "spdsolv_config.h"
#include "spdsolv_setup.h"
#include "spdsolv_main.h"
#include "spdsolv_check.h"
#include "fptype.h"
#include "matfprint.h"
#include "file_log.h"
int m; 
int n; 
int b; 
int reps; 
int check; 
fp_t *A;
fp_t *B;
fp_t *X;
fp_t **Ah;
fp_t *Aorig;	
log_f logf_spd;
int main(int argc, char* argv[]) {
if ( spdsolv_config(argc, argv) ) {
return 1;
}
if ( spdsolv_setup(check, m, n, b, &A, &B, &X) ) {
fprintf(stderr, "err: allocating matrix\n");
return 2;
}
logf_init(&logf_spd, reps, "spdsolv");
int r;
for (r = 0; r < reps; r++) {
logf_record_stime(&logf_spd);
SPDSOLV_MAIN(m, n, b, A, B);
#pragma omp taskwait
logf_record_etime(&logf_spd);
}
int ret = spdsolv_check(check, m, n, X, B);
logf_dump(&logf_spd);
spdsolv_shutdown(A, B, X);
return ret;
}
