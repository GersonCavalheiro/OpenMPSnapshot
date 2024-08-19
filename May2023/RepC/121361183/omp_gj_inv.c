#include<stdio.h>
#include<omp.h>
#include<stdlib.h>
int omp_gj_inv(double **mat,double **inv,long dim,long thread)
{
long k,i,j;
double temp;
omp_set_num_threads(thread);
for(i=0;i<dim;i++) inv[i][i]=1.0;
for(k=0;k<dim;k++)
{
#pragma omp parallel
{
#pragma omp for private(j,temp)
for(i=0;i<dim;i++)
{
if(k!=i)
{
temp=mat[i][k]/mat[k][k];
for(j=k+1;j<dim;j++) mat[i][j]-=temp*mat[k][j];
for(j=0;j<dim;j++) inv[i][j]-=temp*inv[k][j];
}
}
#pragma omp for
for(j=0;j<dim;j++)
{
inv[k][j]=inv[k][j]/mat[k][k];
if(j!=k)
mat[k][j]=mat[k][j]/mat[k][k];
}
mat[k][k]=1.0;
}
}
return 0;
}
