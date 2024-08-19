#include<parallel/parallel-omp.h>
int gauss_fast_omp(int dim,int thread,double **mat,double *x,double *libre)
{
int s;	
int nr;	
int dernierre=0;	
int i,j,k,l;
int tid;		
int *counters;
int proccount,proccount1,replay;
omp_lock_t *cond_m;
omp_set_num_threads(thread);
cond_m=(omp_lock_t *)calloc(dim,sizeof(omp_lock_t));
counters=(int *)calloc(dim,sizeof(int));
#pragma omp for
for(i=0;i<dim;i++) 
{
omp_init_lock(&cond_m[i]);
omp_set_lock(&cond_m[i]);
}
#pragma omp parallel private(i,j,k,l,s,nr,tid,replay,proccount,proccount1,dernierre)
{
tid=omp_get_thread_num();
dernierre=0;
nr=dim%thread;
s=(dim-nr)/thread;
if((tid+1)<nr) s++;
else if((tid+1)==nr)
{
s++;
dernierre=1;
}
if((nr==0) && ((tid+1)==thread)) dernierre=1;
if(tid==0)
{
for(j=1;j<dim;j++) mat[0][j]=mat[0][j]/mat[0][0];
libre[0]=libre[0]/mat[0][0];
mat[0][0]=1.0;
#pragma omp atomic
counters[0]++;
omp_unset_lock(&cond_m[0]);
proccount=proccount1=tid+thread;
#pragma omp barrier
}
else
{
#pragma omp barrier
proccount=proccount1=tid;
}
replay=0;
for(i=proccount;i<dim;i+=thread)
{
for(k=replay;k<i;k++)
{
omp_set_lock(&cond_m[k]);
omp_unset_lock(&cond_m[k]);
for(l=proccount1;l<dim;l+=thread)
{
for(j=(k+1);j<dim;j++)
mat[l][j]-=mat[l][k]*mat[k][j];
libre[l]-=mat[l][k]*libre[k];
mat[l][k]=0;
}
}
if(counters[i-1]==1)
{
for(j=i+1;j<dim;j++)
mat[i][j]=mat[i][j]/mat[i][i];
libre[i]=libre[i]/mat[i][i];
mat[i][i]=1.0;
replay=i;
proccount1=proccount1+thread;
#pragma omp atomic
counters[i]++;
omp_unset_lock(&cond_m[i]);
}
}
#pragma omp barrier
#pragma omp for
for(i=0;i<dim;i++)
{
counters[i]=0;
omp_set_lock(&cond_m[i]);
}
if(dernierre==1)
{
proccount=proccount1=dim-1-thread;
counters[dim-1]=1;
omp_unset_lock(&cond_m[dim-1]);
}
else proccount=proccount1=tid+thread*(s-1);	
#pragma omp barrier
replay=dim-1;
for(i=proccount;i>=tid;i-=thread)
{
for(k=replay;k>i;k--)
{
omp_set_lock(&cond_m[k]);
omp_unset_lock(&cond_m[k]);
for(l=proccount1;l>=tid;l-=thread)
libre[l]-=libre[k]*mat[l][k];
}
if(counters[i+1]==1)
{
counters[i]=1;
replay=i;
proccount1-=thread;
omp_unset_lock(&cond_m[i]);
}
}
}
#pragma omp barrier
#pragma omp for
for(i=0;i<dim;i++)
omp_destroy_lock(&cond_m[i]);
#pragma omp for
for(i=0;i<dim;i++)
x[i]=libre[i];
free(cond_m);
free(counters);
}
