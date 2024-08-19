
#include<solve-openmp.h>


int mul_striped_fast(long dim,int thread,double **mat,double **mat2)
{

long i,j,k;

#pragma omp for private(j)
for(i=0;i<dim;i++)
{
for(j=0;j<dim;j++) mat2[i][j]=mat[i][j];
}
for(i=0;i<3;i++)
{
#pragma omp for private(k)
for(j=0;j<dim;j++)
{
for(k=0;k<dim;k++) mat2[j][k]=mat2[j][k]*mat[j][k];
}
}
return(0);
}
