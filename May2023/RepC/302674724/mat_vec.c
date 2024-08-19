#include <stdio.h>
#include <stdlib.h>
#include <math.h>
void mat_vec(double* a, double* x, double* y, int nn, int nt)
{
#pragma omp parallel num_threads(nt) 
#pragma omp parallel default(none) firstprivate(a,x,y,nn)
{
register int k=0;
register int n=nn;
register int i;
register int j;
#pragma omp for
for(i=0;i<n;i+=4){
register double ty1 = 0;
register double ty2 = 0;
register double ty3 = 0;
register double ty4 = 0;
for(j=0;j<n;j+=2){
register double t0=x[j];
register double t1=x[j+1];
register int n2=2*n;
register int n3=3*n;
k= i*n+j;
ty1  +=a[k]*t0    +a[k+1]*t1  ;
ty2+=a[k+n]*t0+a[k+1+n]*t1;
ty3+=a[k+n2]*t0+a[k+1+n2]*t1;
ty4+=a[k+n3]*t0+a[k+1+n3]*t1;
}
y[i]  = ty1;
y[i+1]+=ty2;
y[i+2]+=ty3;
y[i+3]+=ty4;
}
}  
}
