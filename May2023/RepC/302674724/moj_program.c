#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "pomiar_czasu.h"
#define WYMIAR 10000
#define ROZMIAR 100000000
void mat_vec(double* a, double* x, double* y, int n, int nt);
main ()
{
static double x[WYMIAR],y[WYMIAR],z[WYMIAR];
double *a;
double t1;
int n,nt,i,j;
const int ione=1;
const double done=1.0;
const double dzero=0.0;
a = (double *) malloc(ROZMIAR*sizeof(double));
for(i=0;i<ROZMIAR;i++) a[i]=0.00000001*i;
for(i=0;i<WYMIAR;i++) x[i]=0.0001*(WYMIAR-i);
n=WYMIAR;
nt=4;
printf("\nPoczatek procedury macierz-wektor\n\n");
inicjuj_czas();
t1 = omp_get_wtime();
mat_vec(a,x,y,n,nt);
t1 = omp_get_wtime() - t1;
drukuj_czas();
printf("\nKoniec procedury macierz-wektor\n");
printf("\tczas wykonania: %lf, Gflop/s: %lf, GB/s> %lf\n",  
t1, 2.0e-9*ROZMIAR/t1, (1.0+1.0/n)*8.0e-9*ROZMIAR/t1);
printf("\nPoczatek procedur sprawdzajacych\n");
inicjuj_czas();
t1 = omp_get_wtime();
#pragma omp parallel for num_threads(nt) firstprivate(n) private(j) 
for(i=0;i<n;i++){
double t=0.0;
int ni = n*i;
for(j=0;j<n;j++){
t+=a[ni+j]*x[j];
}
z[i]=t;
}
t1 = omp_get_wtime() - t1;
drukuj_czas();
printf("\nKoniec procedury macierz-wektor\n");
printf("\tczas wykonania: %lf, Gflop/s: %lf, GB/s> %lf\n",  
t1, 2.0e-9*ROZMIAR/t1, (1.0+1.0/n)*8.0e-9*ROZMIAR/t1);
for(i=0;i<WYMIAR;i++){
if(fabs(y[i]-z[i])>1.e-9*z[i]) printf("Blad!\n");
}
}
