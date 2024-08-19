#define NRANSI


#include "nrutil.h"
#include <stdio.h>
#include <pthread.h>
#define MAX_THREADS 10

#define TOL 1.0e-12

int summer_shared_index;
pthread_mutex_t max_mutex, thresh_mutex, summ_mutex;


struct summ_args { 
int ma; 
double *a, *afunc, *sum; 
};


void *dataworker() {

}

void *summer(void * packed_args) {	
struct summ_args *args = (struct summ_args *)packed_args;
int ma = args->ma;
double *a = args->a, *afunc = args->afunc, *sum = args->sum;	
int local_index; 
double partial_sum = 0;
do {
pthread_mutex_lock(&summ_mutex);
local_index = summer_shared_index;
summer_shared_index++;
pthread_mutex_unlock(&summ_mutex);

if (local_index <= ma) {
partial_sum += (*(a+local_index)) * (*(afunc+local_index));
}
} while (local_index <= ma);

pthread_mutex_lock(&summ_mutex);
*sum += partial_sum;
pthread_mutex_unlock(&summ_mutex);

return 0;
}

void svdfit_d(double x[], double y[], double sig[], int ndata, double a[], int ma,
double **u, double **v, double w[], double *chisq,
void (*funcs)(double, double [], int))
{
void svbksb_d(double **u, double w[], double **v, int m, int n, double b[], double x[]);
void svdcmp_d(double **a, int m, int n, double w[], double **v);
int j,i,k;
double wmax,tmp,thresh,sum,*b,*afunc;

b=dvector(1,ndata);
afunc=dvector(1,ma);

#pragma omp parallel for if(ndata > 100)\
shared(funcs, x, afunc, ma, u, sig, b, y) \
private(i, j)
for (i=1;i<=ndata;i++) {	
(*funcs)(x[i],afunc,ma);
for (j=1;j<=ma;j++) u[i][j]=afunc[j]/sig[i];
b[i]=y[i]/sig[i];
}
svdcmp_d(u,ndata,ma,w,v);

wmax=0.0;
double *wptr = &(w[1]);
for (j=1;j<=ma;j++, wptr++)
if (*(wptr) > wmax) wmax=*(wptr);	

thresh=TOL*wmax;
wptr = &(w[1]);
for (j=1;j<=ma;j++,wptr++)
if (*(wptr) < thresh) *(wptr)=0.0;

svbksb_d(u,w,v,ndata,ma,b,a);

*chisq=0.0;
#pragma omp parallel for if(ndata > 100)\
shared(funcs, x, afunc, ma, a, chisq, y, sig) \
private(i, j, sum, tmp)
for (i=1; i<=ndata; i++) {
(*funcs)(x[i],afunc,ma);
for (sum=0.0,j=1;j<=ma;j++) sum += a[j] * afunc[j];


*chisq += (tmp=(y[i]-sum)/sig[i], tmp*tmp);
}
free_dvector(afunc,1,ma);
free_dvector(b,1,ndata);
}


#undef TOL
#undef NRANSI

