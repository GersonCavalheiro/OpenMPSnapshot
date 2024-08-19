#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>
#include <math.h>
#include "timeprint.c"
#define numar 10
int main(int argc,char **argv)
{
struct timeval t1,t2;
long i,l,j,k;
long dim;
int thread;
FILE *fp;
double **mat,*x,*rez,*mtemp1,temp,*y,**mat1,*y1;
dim=atol(argv[1]);
thread=atoi(argv[2]);
x=(double *)calloc(dim,sizeof(double));
y=(double *)calloc(dim,sizeof(double));
y1=(double *)calloc(dim,sizeof(double));
rez=(double *)calloc(dim,sizeof(double));
for(i=0;i<dim;i++) rez[i]=(double)i;
mat=(double **)calloc(dim,sizeof(double *));
mtemp1=(double *)calloc(dim*dim,sizeof(double));
for(i=0;i<dim;i++)
{
mat[i]=mtemp1;
mtemp1+=dim;
}
mat1=(double **)calloc(dim,sizeof(double *));
mtemp1=(double *)calloc(dim*dim,sizeof(double));
for(i=0;i<dim;i++)
{
mat1[i]=mtemp1;
mtemp1+=dim;
}
for(i=0;i<dim;i++)
{
for(j=0;j<dim;j++) mat[i][j]=(double)rand();
temp=0.0;
for(j=0;j<dim;j++) if(j!=i) temp+=fabs(mat[i][j]);
mat[i][i]+=temp;
}
for(i=0;i<dim;i++)
{
y[i]=0.0;
x[i]=0.0;
for(j=0;j<dim;j++) y[i]+=mat[i][j]*rez[j];
}
gettimeofday(&t1,NULL);
for(l=0;l<numar;l++)
{	
#pragma omp parallel for private(j)
for(i=0;i<dim;i++)
for(j=0;j<dim;j++) mat1[i][j]=mat[i][j];
#pragma omp parallel for
for(i=0;i<dim;i++) y1[i]=y[i];
omp_set_num_threads(thread);
for(k=0;k<dim;k++)
{
#pragma omp parallel for private(j,temp)
for(i=0;i<dim;i++)
{
if(i!=k)
{
temp=mat1[i][k]/mat1[k][k];
for(j=k+1;j<dim;j++) mat1[i][j]-=temp*mat1[k][j];
y1[i]-=temp*y1[k];
}
}
}
#pragma omp parallel for
for(i=0;i<dim;i++) x[i]=y1[i]/mat1[i][i];
}
gettimeofday(&t2,NULL);
for(i=0;i<dim;i++) {if(fabs(rez[i]-x[i])>1E-5) printf("%lf=%lf\n",rez[i],x[i]); fflush(stdout);}
fp=fopen("time-openmp.dat","a");
timeprint(t1,t2,numar,dim,fp,thread);
fclose(fp);
free(*mat);
free(mat);		
free(*mat1);
free(mat1);
free(x);
free(rez);
free(y);
free(y1);
return 0;
}
