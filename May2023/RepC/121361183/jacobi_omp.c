#include<string.h>
#include<math.h>
#include<omp.h>
void jacobi_omp(double **mat,double *ty,double *tx,int dim,double err,int thread)
{
double *xn_1;
long i,j;
int th;
double q,sum,temp;
double *sum_p;
xn_1=(double *)calloc(dim,sizeof(double));
sum_p=(double *)calloc(thread,sizeof(double));
q=0.0;
omp_set_num_threads(thread);
#pragma omp parallel private(th,i)
{
#pragma omp for
for(i=0;i<dim;i++)
tx[i]=ty[i]/mat[i][i];
#pragma omp for reduction(+:q)
for(i=1;i<dim;i++)
q+=fabs(mat[0][i]/mat[0][0]);
th=omp_get_thread_num();
sum_p[th]=q;
#pragma omp for private(temp,j)
for(i=1;i<dim;i++)
{
temp=0.0;
for(j=0;j<dim;j++)
if(i!=j) temp+=fabs(mat[i][j]/mat[i][i]);
if(sum_p[th]<temp) sum_p[th]=temp;
}
#pragma omp single
{
q=sum_p[0];
for(i=1;i<thread;i++)
if(q<sum_p[i])
q=sum_p[i];
}
sum_p[th]=fabs(ty[th]/mat[th][th]);
for(i=th+thread;i<dim;i=i+thread)
{
if(sum_p[th]<fabs(ty[i]/mat[i][i]))
sum_p[th]=fabs(ty[i]/mat[i][i]);
}
#pragma omp barrier
#pragma omp single
{
sum=sum_p[0];
for(i=1;i<thread;i++)
if(sum<sum_p[i]) sum=sum_p[i];
sum=sum*q/(1-q);
}
while(fabs(sum)>err)
{	
#pragma omp single
memcpy(xn_1,tx,dim*sizeof(double));
#pragma omp for private(j)
for(i=0;i<dim;i++)
{
tx[i]=ty[i]/mat[i][i];
for(j=0;j<dim;j++)
if(j!=i) tx[i]-=mat[i][j]/mat[i][i]*xn_1[j];
}
sum_p[th]=fabs(tx[th]-xn_1[th]);
for(i=th+thread;i<dim;i=i+thread)
if(sum_p[th]<fabs(tx[i]-xn_1[i])) sum_p[th]=fabs(tx[i]-xn_1[i]);
#pragma omp barrier
#pragma omp single
{
sum=sum_p[0];
for(i=1;i<thread;i++)
if(sum<sum_p[i]) sum=sum_p[i];
sum=sum*q/(1-q);
}
}
}
free(xn_1);
}
