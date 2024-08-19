#include<stdio.h>
#include<omp.h>
int main()
{
int nthreads,tid;
int i,j,cond;
omp_lock_t lock;
omp_set_num_threads(10);
#pragma omp parallel shared(lock,cond) private(tid,i)
{
tid=omp_get_thread_num();
if(tid==0)
{
cond=0;
i=0;
#pragma omp barrier
while(i!=1)
{
printf("tid=0\n");fflush(stdout);
scanf("%d",&i);fflush(stdin);
#pragma omp atomic 
cond=cond+3;
}
#pragma omp atomic
cond=cond+10;
printf("Tid=%d\n",tid);fflush(stdout);
}else
if((tid==1)||(tid==2)||(tid==3))
{
#pragma omp barrier
while(cond!=10)
{
if(cond==tid)
{
#pragma omp atomic
cond--;
printf("Tidr=%d\n",tid);fflush(stdout);
}
}
}else
{
#pragma omp barrier
printf("Tid=%d\n",tid);fflush(stdout);
}
}
}