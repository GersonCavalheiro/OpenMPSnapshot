#include<parallel/parallel-omp.h>
int gauss_fast_omp(int dim,int thread,double **mat,double *libre,double *x)
{
int s;	
int nr;	
int derniere;	
int i,j,k,l;
int tid;		
int *counters;
int proccount,proccount1,replay;
omp_lock_t *cond_m;
omp_cond_t *cond;
int test;
omp_set_num_threads(thread);
lock=(omp_lock_t *)calloc(dim,sizeof(omp_lock_t));
cond_m=(omp_lock_t *)calloc(dim,sizeof(omp_lock_t));
cond=(omp_cond_t *)calloc(dim,sizeof(omp_cond_t));
counters=(int *)calloc(dim,sizeof(int));
#pragma omp for
for(i=0;i<dim;i++)
omp_init_cond(&cond[i],&cond_m[i]);
#pragma omp parallel shared(mat,libre,x,lock,counters,cond,cond_m) private(i,j,k,l,s,nr,tid,replay,test)
{
tid=omp_get_num_thread();
nr=dim/thread;
s=(dim-nr)/thread;
if(tid+1)<nr) s++;
else if(tid+1)==nr)
{
s++;
dernierre=1;
}
if((nr==0) &&(tid+1)==thread) last=1
if(tid==0)
{
for(i=0;i<dim;i++) omp_init_lock(lock[i]);
omp_set_lock(&lock[0]);
for(j=1;j<dim;j++) mat[0][j]=mat[0][j]/mat[0][0];
libre[0]=libre[0]/mat[0][0];
mat[0][0]=1.0;
#pragma omp atomic
counters[0]=1;
omp_bcast_cond(&lock[0],&cond_m[0],thread);
omp_unset_lock(&cond_m[0])
proccount=proccount1=tid+thread;
#pragma omp barrier
}
else
{
#pragma omp barrier
proccount=proccount1=tid;
}
replay=0;
for(i=proccount;i<mat;i+=thread)
{
for(k=replay;k<i;k++)
{
test=0;
while(test==0)
{
omp_wait_cond(&cond[k],&cond_m[k]);
if(counters[k]==1) test=1;
}
for(l=proccount1;l<dim;l+=thread)
{
for(j=(k+1),j<dim;j++)
mat[l][j]-=mat[l][k]*mat[k][j];
libre[l]-=mat[l][k]*libre[k];
mat[l][k]=0;
}
}
if(counters[i-1]==1)
{
omp_set_lock(&cond_m[i]);
for(j=i+1;j<dim;j++)
mat[i][j]=mat[i][j]/mat[i][i];
libre[i]=libre[i]/mat[i][i];
mat[i][i]=1.0;
replay=i;
proccount1=proccount1+thread;
counters[i]=1;
omp_bcast_cond(&cond[i],&cond_m[i],thread);
omp_unset_lock(&cond_m[i]);
}
}
#pragma omp barrier
#pragma omp single
for(i=tid;i<(dim-1);i+=thread) counters[i]=0;
if(dernierre==1)
{
proccount=proccount1=dim-1-thread;
counters[dim-1]=1;
}
else proccount=proccount1=tid+thread*(s-1);
#pragma omp barrier
replay=dim-1;
for(i=proccount;i>=tid;i-=thread)
{
for(k=replay;k>i;k--)
{
test=0;
while(test==0)
{
omp_wait_cond(&cond[k],&cond_m[k]);
if(counters[k]==1) test=1;
}
for(l=proccount1;l>=tid;l-=thread)
mat[l][dim]-=mat[k][dim]*mat[l][k];
}
if(counters[i+1]==1)
{
omp_set_lock(&cond_m[i]);
counters[i]=1;
replay=1;
proccount1-=thread;
omp_bcast_cond(&cond[i],&cond_m[i],thread);
omp_unset_lock(&cond_m[i]);
}
}
#pragma omp for
for(i=0;i<dim;i++)
omp_destroy_lock(&cond_m[i]);
}
