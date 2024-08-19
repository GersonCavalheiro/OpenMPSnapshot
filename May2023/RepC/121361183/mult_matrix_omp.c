#include<parallel/parallel-omp.h>
void mult_sq_mat_striped_omp(int mat,int thread,double **a,double **b,double **c,int type)
{
int i,j,k;
int tid;
omp_set_num_threads(thread);
#pragma omp parallel for shared(a,b,c) private(i,j,k,tid)
{
for(i=0;i<mat;i++)
for(j=0;j<mat;j++)
{
c[i][j]=0.0;
for(k=0;k<mat;k++)
c[i][j]+=a[i][k]*b[k][j];
}
}
}
void mult_sq_mat_striped_omp_1(int mat,int thread,double **a,double **b,double **c,int type)
{
int i,j,k;
int tid;
omp_set_num_threads(thread);
#pragma omp parallel shared(a,b,c) private(i,j,k,tid)
{
tid=omp_get_thread_num();
for(i=tid;i<mat;i=i+thread)
for(j=0;j<mat;j++)
{
c[i][j]=0.0;
for(k=0;k<mat;k++)
c[i][j]+=a[i][k]*b[k][j];
}
}
}
void mult_sq_mat_check_omp(int mat,int thread,double **a,double **b,double **c,int type)
{
int i,j,k,l,m;
int q,s,temp;
int limite0,limite1;
q=sqrt(thread);
s=(int)mat/q;
temp=mat-s*q;
omp_set_num_threads(thread);
#pragma omp parallel for shared(a,b,c,s,q) private(i,j,k,l,m)
{
for(l=0;l<q;l++)
{		
#pragma omp parallel for shared(a,b,c,s,q) private(i,j,k,m)
{
for(m=0;m<q;m++)
{
if(l==q-1) limite0=mat;
else limite0=s*(l+1);
if(m==q-1) limite1=mat;
else limite1=s*(m+1);
for(i=s*l;i<limite0;i++)
for(j=s*m;j<limite1;j++)
{
c[i][j]=0.0;
for(k=0;k<mat;k++)
c[i][j]+=a[i][k]*b[k][j];
}
}
}
}
}
}