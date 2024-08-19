#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>
#include "cg_main.h"
#include "cgmod1_main.h"
#include "cgpipe_main.h"
#include "cgprof_main.h"
#include "cg_config.h"
#include "cg_setup.h"
#include "fptype.h"
#include "ompss_apps.h"
#include "ompss_options.h"
int n; 
int bm; 
int bn; 
int s; 
int it; 
int lookahead; 
int async;
double prec; 
int rep; 
fp_t *x; 
fp_t *rhs;  
fp_t *xstar; 
fp_t *work; 
int warmup;
char *rhsfname;
unsigned long works;
double profile;
void *A;
void *Ahhb;
int main (int argc, char *argv[]) 
{
if ( cg_config(argc, argv) ) {
return 1;
}
if ( cg_setup(n, bm, bn, s, &A, &x, &rhs, &xstar, &work) ) {
return 2;
}
int cgprof = ompssopt_read(ompssapp_CG, "log", OMPSSOPT_NO);
unsigned long elapsed = 0;
int r;
for ( r=0; r<rep; ++r) {
x = memset(x, 0, n*s*sizeof(fp_t));
struct timeval start;
gettimeofday(&start, NULL);
int offs;
A = Ahhb;
#if USE_CG 
CG(bm, bn, n, A, s, rhs, x, &offs, prec, it, work, works, lookahead, async, profile, cgprof);
#elif USE_CGPROF
CGPROF(bm, bn, n, A, s, rhs, x, &offs, prec, it, work, works, lookahead, async, profile, warmup, cgprof, 1, 1E3);
#elif USE_CGMOD1 
CGMOD1(bm, bn, n, A, s, rhs, x, &offs, prec, it, work, works, lookahead, async, profile, warmup, cgprof, 1);
#elif USE_CGPIPE
CGPIPE(bm, bn, n, A, s, rhs, x, &offs, prec, it, work, works, lookahead, async, profile, warmup, cgprof);
#endif
#pragma omp taskwait
struct timeval stop;
gettimeofday(&stop, NULL);
unsigned long itlaps = (stop.tv_sec - start.tv_sec) * 1e6 + stop.tv_usec - start.tv_usec;
elapsed += itlaps;
}
#if 1
double flelapsed=(double)elapsed;
FILE *f = fopen("ompss.log", "w");
fprintf(f, "time : %.2f\n", flelapsed / (double) rep);
fclose(f);
#endif
cg_cleanup(n, s, A, x, rhs, xstar, work);
return 0;
}
