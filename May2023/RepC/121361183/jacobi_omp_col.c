#include<string.h>
#include<math.h>
#include<omp.h>
void jacobi_omp_col(double **mat,double *ty,double *tx,long dim,double err,int thread)
{
double *xn_1;
double *yn,*yn_1;
double max,sum,q;
long i,j;
double count;
int th;
double *sum_p;
xn_1=(double *)calloc(dim,sizeof(double));
yn=(double *)calloc(dim,sizeof(double));
yn_1=(double *)calloc(dim,sizeof(double));
sum_p=(double *)calloc(thread,sizeof(double));
q=0.0;
omp_set_num_threads(thread);
#pragma omp parallel private(th,i)
{
#pragma omp for reduction(+:q)
for(i=1;i<dim;i++)
q+=fabs(mat[i][0]/mat[i][i]);
th=omp_get_thread_num();
sum_p[th]=q;
#pragma omp for private(sum,j)
for(i=1;i<dim;i++)
{
sum=0.0;
for(j=0;j<dim;j++) if(i!=j) sum+=fabs(mat[j][i]/mat[j][j]);
if(sum_p[th]<sum) sum_p[th]=sum;
}
#pragma omp single
{
q=sum_p[0];
for(i=1;i<thread;i++)
if(q<sum_p[i]) q=sum_p[i];
}
sum_p[th]=fabs(mat[th][th]);
for(i=th+thread;i<dim;i=i+thread)
if(sum_p[th]>fabs(mat[i][i])) sum_p[th]=fabs(mat[i][i]);
#pragma omp barrier
#pragma omp single
{
max=sum_p[0];
for(i=1;i<thread;i++)
if(max>sum_p[i]) max=sum_p[i];
count=q/(max*(1-q));
memcpy(yn,ty,dim*sizeof(double));  	
sum=0.0;
}
#pragma omp for reduction(+:sum)
for(i=0;i<dim;i++)
sum+=fabs(yn[i]);
#pragma omp single 
count=count*sum;
while(fabs(count)>err)
{
#pragma omp single
memcpy(yn_1,yn,dim*sizeof(double));
#pragma omp for private(j)
for(i=0;i<dim;i++)
{
yn[i]=ty[i];
for(j=0;j<dim;j++) if(i!=j) yn[i]-=mat[i][j]/mat[j][j]*yn_1[j];
tx[i]=yn[i]/mat[i][i];
}
sum=0.0;
#pragma omp for reduction(+:sum)
for(i=0;i<dim;i++)
sum+=fabs(yn[i]-yn_1[i]);
#pragma omp single
count=q*sum/(max*(1-q));
}
}
free(xn_1);
free(yn);
free(yn_1);
free(sum_p);
}
