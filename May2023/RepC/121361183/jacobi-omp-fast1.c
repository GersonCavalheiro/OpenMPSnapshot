#include<solve-omp.h>
int jacobi_fast_omp(long dim,int thread,double **mat,double *x,double *y,double eroare)
{
long i,j;
double *oldx;
double temp;
int flag;
omp_set_num_threads(thread);
oldx=(double *)calloc(dim,sizeof(double));
#pragma omp parallel for
for(i=0;i<dim;i++) for(j=0;j<dim;j++) x[i]=y[i];
flag=0;
#pragma omp parallel private(temp)
while(flag!=dim)
{
flag=0;
#pragma omp for
for(i=0;i<dim;i++) oldx[i]=x[i];
#pragma omp barrier
#pragma omp for
for(i=0;i<dim;i++)
{
temp=0.0;
for(j=0;j<dim;j++) if(j!=i) temp+=mat[i][j]*oldx[j];
x[i]=(y[i]-temp)/mat[i][i];
if(abs(x[i]-oldx[i])<=eroare)
{
#pragma omp atomic
flag++;
}
}
}
free(oldx);
return 0;
}
