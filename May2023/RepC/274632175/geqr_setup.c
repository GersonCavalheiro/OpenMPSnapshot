#include "geqr_setup.h"
#include <stdlib.h>
#include <time.h>
#include "matfprint.h"
#include "densutil.h"
#include "genmat.h"
static double *A;
static double *T;
static double *S;
int geqr_setup(int check, int m, int mr, int n, int nr, int tr, int tc, int bs, int mt, int nt, int mleft, int nleft, double ***Ah, double ***Th, double ***Sh, double **Aorig)
{
*Ah = (double **) malloc( mt * nt * sizeof(double *) );
if(*Ah == NULL) return 1;
*Th = (double **) malloc( mt * nt * sizeof(double *) );
if(*Th == NULL) return 1;
*Sh = (double **) malloc( mt * nt * sizeof(double *) );
if(*Sh == NULL) return 1;
double **lAh = *Ah;
double **lTh = *Th;
double **lSh = *Sh;
A = GENMAT(m, nr, mr);
#pragma omp register ([mr*nr]A)
T = (double *) malloc( mr * nr * sizeof(double) );
S = (double *) malloc( bs * mt * nr *sizeof(double) );
int j;
for (j = 0; j < nt; j++) {
int i;
for (i = 0; i < mt; i++) {
lAh[j*mt+i] = (double *) &A[j*tc*mr+i*tr*tc];
}
}
for (j = 0; j < nt; j++) {
int i;
for (i = 0; i < mt; i++) {
lTh[j*mt+i] = &T[j*mr+i*tc];
}
}
for (j = 0; j < nt; j++) {
int i;
for (i = 0; i < mt; i++) {
lSh[j*mt+i] = &S[j*bs*mt*tc+i*bs*tc];
}
}
*Aorig=NULL;
if(check) {
*Aorig = CMH2CM(m, n, tr, tc, lAh[0], mr );
}
return 0;
}
void geqr_shutdown()
{
free(A);
free(T);
free(S);
}
