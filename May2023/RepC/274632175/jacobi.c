#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include "jacobi_config.h"
#include "jacobi_setup.h"
#include "jacobi_main.h"
#include "jacobi_check.h"
#include "densutil.h"
#include "vector.h"
#include "bsblas_gemv_csr.h"
#include "file_log.h"
void *Ahb;
void *Ahbh;
fp_t *v_b; 
fp_t *v_x; 
fp_t *v_x0;
int format; 
int bs; 
int dim; 
int max_iter;
hbmat_t **diagL; 
hbmat_t **A1;
int lookahead;
fp_t threshold;
char *fname;
int check;
int rep;
fp_t *work;
log_f logf_jac;
int res_p;
int main(int argc, char* argv[])
{
if ( jacobi_config(argc, argv) != 0 ) {
return 1;
}
if ( jacobi_setup(fname) != 0 ) {
return 2;
}
logf_init(&logf_jac, rep, "jacobi");
unsigned int iter;
int r;
for ( r = 0; r < rep; ++r ) {
logf_record_stime(&logf_jac);
iter = JACOBI_MAIN_CSR(Ahbh, v_x0, v_b, bs, max_iter, diagL, lookahead, threshold, work, &res_p);
#pragma omp taskwait
logf_record_etime(&logf_jac);
}
fp_t res = jacobi_check(check, res_p);
logf_dump(&logf_jac);
jacobi_shutdown();
return 0;
}
