#include<parallel/parallel-omp.h>
int mult_sq_mat_striped_omp(int mat,double **a,double **b,double **c)
{
int i,j,k;
#pragma omp parallel for private(i,j,k,tid)
for(i=0;i<mat;i++)
for(j=0;j<mat;j++)
{
c[i][j]=0.0;
for(k=0;k<mat;k++)
c[i][j]+=a[i][k]*b[k][j];
}
return omp_get_num_threads();
}
