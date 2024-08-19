#include<solve-omp.h>
int gauss_fast_omp(long dim,int thread,double **mat,double *x,double *libre)
{
long s;	
long nr;	
int dernierre=0;	
long i,j,k,l;
int tid;		
int *counters;
long proccount,proccount1,replay;
int test;
omp_lock_t *cond_m;
omp_set_num_threads(thread);
cond_m=(omp_lock_t *)calloc(dim,sizeof(omp_lock_t));
counters=(int *)calloc(dim,sizeof(int));
#pragma omp parallel for
for(i=0;i<dim;i++) 
{
omp_init_lock(&cond_m[i]);
omp_set_lock(&cond_m[i]);
}
#pragma omp parallel private(i,j,k,l,s,nr,tid,replay,proccount,proccount1,dernierre,test)
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
}
else
proccount=proccount1=tid;
replay=0;
test=0;
for(i=proccount;i<dim;i+=thread)
{
proccount1=i+thread;
for(k=replay;k<i;k++)
{
while(replay<k && counters[k]==0 && proccount1<dim)
{
for(j=replay+1;j<dim;j++)
mat[proccount1][j]-=mat[proccount1][replay]*mat[replay][j];
libre[proccount1]-=mat[proccount1][replay]*libre[replay];
mat[proccount1][replay]=0.0;
proccount1+=thread;
if(proccount1>=dim)
{
proccount1=i+thread;
test=0;
replay++;
}
else test=1;
}
if(counters[k]==0)
{
omp_set_lock(&cond_m[k]);
omp_unset_lock(&cond_m[k]);
}
for(j=k+1;j<dim;j++)
mat[i][j]-=mat[i][k]*mat[k][j];
libre[i]-=mat[i][k]*libre[k];
mat[i][k]=0.0;
}
for(j=i+1;j<dim;j++)
mat[i][j]=mat[i][j]/mat[i][i];
libre[i]=libre[i]/mat[i][i];
mat[i][i]=1.0;
#pragma omp atomic
counters[i]++;
omp_unset_lock(&cond_m[i]);
for(k=replay;k<i;k++)
{
for(l=proccount1;l<dim;l+=thread)
{
for(j=k+1;j<dim;j++)
mat[l][j]-=mat[l][k]*mat[k][j];
libre[l]-=mat[l][k]*libre[k];
mat[l][k]=0.0;
}
if(test==1)
{
proccount1=i+thread;
test=0;
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
if(dernierre==1)
{
#pragma omp atomic
counters[dim-1]++;
omp_unset_lock(&cond_m[dim-1]);
proccount=proccount1=dim-1-thread;
}
else proccount=proccount1=tid+thread*(s-1);	
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
libre[i]-=libre[k]*mat[i][k];
}
#pragma omp atomic
counters[i]++;
omp_unset_lock(&cond_m[i]);
proccount1=i-thread;
for(k=replay;k>i;k--)
{
for(l=proccount1;l>=tid;l-=thread)
libre[l]-=libre[k]*mat[l][k];
}
replay=i;
}
}
#pragma omp barrier
#pragma omp parallel for
for(i=0;i<dim;i++)
omp_destroy_lock(&cond_m[i]);
#pragma omp parallel for
for(i=0;i<dim;i++)
x[i]=libre[i];
free(cond_m);
free(counters);
return 0;
}
