#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <math.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include "qr_config.h"
#include "geqr_setup.h"
#include "qr_check.h"
#include "geqrf_main.h"
int m;
int n;
int tr;
int tc;
int bs;
int reps;
int check;
int main(int argc, char* argv[]) 
{
if ( qr_config(argc, argv) ) {
return 1;
}
int mindim = m < n ? m : n;
int mt = (m+tr-1) / tr; 
int nt = (n+tc-1) / tc; 
int mr = mt * tr; 
int nr = nt * tc; 
int ntleft = nr - n;  
int mtleft = mr - m; 
int diagl = (mindim + tc - 1) / tc;
printf("%i reps A blocks %ix%i elems %ix%i->%ix%i left %i %i steps %i\n", reps, mt, nt, m, n, mr, nr, mtleft, ntleft,diagl);
double **Ah, **Th, **Sh; 
double *Aorig = NULL;
if ( geqr_setup(check, m, mr, n, nr, tr, tc, bs, mt, nt, mtleft, ntleft, &Ah, &Th, &Sh, &Aorig) ) {
return 2;
}
unsigned long elapsed= 0 ;
int r;
for (r = 0; r < reps; r++) {
struct timeval start;
gettimeofday( &start, NULL );
#if 0
#if USE_BINARY_TREE
qrcabin(diagl, tr, tc, bs, mt, nt);
#elif USE_REORDR
qrcareordr(diagl, tr, tc, bs, mt, nt);
#elif USE_NODEPS
qrcarect_nodeps(diagl, tr, tc, bs, mt, nt);
#elif USE_GEQRT
#endif
#endif
geqrf(diagl, tr, tc, bs, mt, nt, Ah, Th, Sh);
#pragma omp taskwait
struct timeval stop;
gettimeofday(&stop,NULL);
unsigned long interm = stop.tv_usec - start.tv_usec;
interm += (stop.tv_sec - start.tv_sec ) * 1000000;
elapsed += interm;
}
int ret = qr_check(check, m, n, tr, tc, Ah, Aorig);
double dn = (double) n;
double elapsedfl = (double) elapsed;
#if 0
printf( "time (usec.):\t%.2f GFLOPS: %.4f\n", elapsedfl / (double)reps, (((4.0/3.0)*dn*dn*dn)/((elapsedfl/(double)reps)*1.0e+3)) );
#else
FILE *statf=fopen("ompss-stats-0001","w");
fprintf(statf,"time : %.2f\n",elapsedfl / (double) reps);
fclose(statf);
#endif
free(Ah);
free(Th);
free(Sh);
free(Aorig);
geqr_shutdown();
return ret;
}
