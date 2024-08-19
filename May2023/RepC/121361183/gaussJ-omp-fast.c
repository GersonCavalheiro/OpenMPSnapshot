#include<solve-omp.h>
int gaussJ_fast_omp(long dim,int thread,double **mat,double *x,double *libre)
{
long s;	
long nr;
int dernierre=0;	
long i,j,k,l;
int tid;		
int *counters;
long proccount,proccount1,replay;
omp_lock_t *cond_m;
double temp;
omp_set_num_threads(thread);
cond_m=(omp_lock_t *)calloc(dim,sizeof(omp_lock_t));
counters=(int *)calloc(dim,sizeof(int));
#pragma omp parallel for
for(i=0;i<dim;i++) 
{
omp_init_lock(&cond_m[i]);
omp_set_lock(&cond_m[i]);
}
#pragma omp parallel private(i,j,k,l,s,nr,tid,replay,proccount,proccount1,dernierre,temp)
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
#pragma omp atomic
counters[0]++;
omp_unset_lock(&cond_m[0]);
proccount=proccount1=tid+thread;
}
else
proccount=proccount1=tid;
replay=0;
for(i=proccount;i<dim;i+=thread)
{
for(k=replay;k<i;k++)
{
if(counters[k]==0)
{
omp_set_lock(&cond_m[k]);
omp_unset_lock(&cond_m[k]);
}
temp=mat[i][k]/mat[k][k];
for(j=(k+1);j<dim;j++)
mat[i][j]-=mat[k][j]*temp;
libre[i]-=libre[k]*temp;
mat[i][k]=0;
}
#pragma omp atomic
counters[i]++;
omp_unset_lock(&cond_m[i]);
proccount1=i+thread;
for(k=replay;k<i;k++)
{
for(l=proccount1;l<dim;l+=thread)
{
temp=mat[l][k]/mat[k][k];
for(j=(k+1);j<dim;j++)
mat[l][j]-=mat[k][j]*temp;
libre[l]-=libre[k]*temp;
mat[l][k]=0;
}
}
replay=i;
}
#pragma omp barrier
#pragma omp for
for(i=0;i<dim;i++)
{
counters[i]=0;
omp_set_lock(&cond_m[i]);
}
#pragma omp barrier
if(dernierre==1)
{
#pragma omp atomic
counters[dim-1]++;
omp_unset_lock(&cond_m[dim-1]);
proccount=proccount1=dim-thread-1;
}
else
proccount=proccount1=tid+thread*(s-1);
replay=dim-1;
for(i=proccount;i>=tid;i-=thread)
{
for(k=replay;k>i;k--)
{
if(counters[k]==0)
{
omp_set_lock(&cond_m[k]);
omp_unset_lock(&cond_m[k]);
}
temp=mat[i][k]/mat[k][k];
for(j=(k+1);j<dim;j++)
mat[i][j]-=mat[k][j]*temp;
libre[i]-=libre[k]*temp;
mat[i][k]=0;
}
#pragma omp atomic
counters[i]++;
omp_unset_lock(&cond_m[i]);
proccount1=i-thread;
for(k=replay;k>i;k--)
{
for(l=proccount1;l>=0;l-=thread)
{
temp=mat[l][k]/mat[k][k];
for(j=(k+1);j<dim;j++)
mat[l][j]-=mat[k][j]*temp;
libre[l]-=libre[k]*temp;
mat[l][k]=0;
}
}
replay=i;
}
}
#pragma omp barrier
#pragma omp parallel for
for(i=0;i<dim;i++)
x[i]=libre[i]/mat[i][i];
#pragma omp parallel for
for(i=0;i<dim;i++)
omp_destroy_lock(&cond_m[i]);
free(cond_m);
free(counters);
return 0;
}
