#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include "fptype.h"
#include "lu_kernels.h"
#include "lu_config.h"
#include "lu_check.h"
#include "lu_setup.h"
#include "lu_main.h"
#include "lull_main.h"
#include "lurecurs_main.h"
#include "file_log.h"
int m;
int n;
int t;
int reps;
int check;
fp_t *A;  
fp_t *Aorig;
int *IPIV;
log_f logf_lu;
int main(int argc, char* argv[])
{
if (lu_config(argc, argv) )
return 1;
lu_setup(&A, &Aorig, &IPIV, m, n, check);
#if USE_LAPACK_STYLE
logf_init(&logf_lu, reps, "lu");
#elif USE_LL
logf_init(&logf_lu, reps, "lull");
#endif
int k;
for (k = 0 ; k < reps ; k++) {
logf_record_stime(&logf_lu);
#if USE_LAPACK_STYLE
LU_MAIN(m, n, t, A, m, IPIV);
#elif USE_LL
LULL_MAIN(m, n, t, A, m, IPIV);
#endif
#pragma omp taskwait
logf_record_etime(&logf_lu);
}
fp_t ret = lu_check(check, m, n, A, Aorig);
logf_dump(&logf_lu);
if (check) {
char tmp[32];
snprintf(tmp, 32, "relerr: %e\n", ret);
logf_append(&logf_lu, tmp);
}
lu_shutdown(A, Aorig, IPIV);
return ret;
}
