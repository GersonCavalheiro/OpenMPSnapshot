#include "gauss-seidel_config.h"
#include "gauss-seidel_setup.h"
#include "gauss-seidel_main.h"
#include "gauss-seidel_check.h"
#include "densutil.h"
#include "vector.h"
#include "array.h"
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
log_f logf_gs;
int main(int argc, char* argv[])
{
if ( gauss_seidel_config(argc, argv) != 0 ) {
return 1;
}
if ( gauss_seidel_setup(fname) != 0 ) {
return 2;
}
logf_init(&logf_gs, rep, "gs");
int p;
int r;
for ( r = 0; r < rep; ++r ) {
array_clear(v_x0, dim);
logf_record_stime(&logf_gs);
GS_MAIN_CSR(Ahbh, v_x0, v_b, bs, max_iter, diagL, lookahead, threshold, work);
#pragma omp taskwait
logf_record_etime(&logf_gs);
gauss_seidel_check(check);
}
logf_dump(&logf_gs);
gauss_seidel_shutdown();
return 0;
}
