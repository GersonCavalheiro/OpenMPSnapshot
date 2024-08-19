#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <pthread.h>
#include "fptype.h"
#include "chol_config.h"
#include "chol_setup.h"
#include "chols_llmain.h"
#include "matfprint.h"
#include "file_log.h"
int m; 
int ts; 
int bs; 
int reps; 
int check; 
int mt; 
int mr; 
int mtleft;
fp_t *A;
fp_t **Ah;
fp_t *Aorig;	
void *Ahb;
void *Acsr;
void *Acsc;
int *work;
int format;
long nzA;
long nzL;
log_f logf_schol;
int main(int argc, char* argv[]) {
if ( chol_config(argc, argv) ) {
return 1;
}
if ( chol_setup(check, m, mr, ts, bs, mt, mtleft) ) {
fprintf(stderr, "err: allocating matrix\n");
return 2;
}
logf_init(&logf_schol, reps, "schols");
unsigned long elapsed= 0 ;
int r;
for (r = 0; r < reps; r++) {
logf_record_stime(&logf_schol);
#if USE_SLL
if ( format == MAT_CSC ) {
chols_ll(Acsc, Acsr, work);
} else {
chols_ll_upper(Acsr, Acsc, work);
}
#endif
#pragma omp taskwait
logf_record_etime(&logf_schol);
}
logf_dump(&logf_schol);
#if USE_SLL
nzL = 0;
hbmat_t* Z = Acsr;
for(int i = 0; i < Z->m; ++i){
for(int j = Z->vptr[i]; j < Z->vptr[i+1]; ++j){
nzL += ((hbmat_t**)Z->vval)[j]->elemc;
}
}
if (nzA > nzL)
nzL = 2 * nzL - Z->m * bs;
char tmp[32];
snprintf(tmp, 32, "degree: %lf\n", ((double)nzL / (double)nzA));
logf_append(&logf_schol, tmp);
#endif
chol_shutdown();
return 0;
}
