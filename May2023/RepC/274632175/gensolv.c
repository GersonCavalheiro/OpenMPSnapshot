#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "gensolv_config.h"
#include "gensolv_setup.h"
#include "gensolv_main.h"
#include "gensolv_check.h"
#include "fptype.h"
#include "matfprint.h"
#include "file_log.h"
int m; 
int n; 
int b; 
int reps; 
int check; 
fp_t *A;
fp_t *Aorig;	
fp_t *B;
fp_t *Borig;
fp_t *X;
fp_t **Ah;
log_f logf_spd;
int main(int argc, char* argv[]) 
{
if ( gensolv_config(argc, argv) ) {
return 1;
}
if ( gensolv_setup(check, m, n, b, &A, &B, &Aorig, &Borig) ) {
fprintf(stderr, "err: allocating matrix\n");
return 2;
}
logf_init(&logf_spd, reps, "gensolv");
int r;
for (r = 0; r < reps; r++) {
logf_record_stime(&logf_spd);
GENSOLV_MAIN(m, n, b, A, B);
#pragma omp taskwait
logf_record_etime(&logf_spd);
}
fp_t ret = gensolv_check(check, m, n, A, B, Aorig, Borig);
logf_dump(&logf_spd);
if (check){
char tmp[32];
snprintf(tmp, 32, "relerr: %e\n", ret);
logf_append(&logf_spd, tmp);
}
gensolv_shutdown(A, B, Aorig, Borig);
return ret;
}
