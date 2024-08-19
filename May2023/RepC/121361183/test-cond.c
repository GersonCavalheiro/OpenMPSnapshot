#include<stdio.h>
#include<omp.h>
#include"cond-omp.h"
int main()
{
int nthreads,tid;
int i,j;
omp_lock_t lock;
omp_cond_t cond;
omp_set_num_threads(10);
i=1;
#pragma omp parallel shared(lock,cond,i) private(tid)
{
tid=omp_get_thread_num();
if(tid==0)
{
omp_init_lock(&lock);
omp_init_cond(&cond,&lock);
#pragma omp barrier
printf("Introduceti\n");fflush(stdout);
while(i==1)
{
scanf("%d",&j);fflush(stdin);
omp_signal_cond(&cond,&lock);
if(j==1)
#pragma omp atomic
i--;
}
printf("Th=%d\n",tid);fflush(stdout);
} else
if((tid==2)||(tid==3)||(tid==8))
{
while(i==1)
{
#pragma omp barrier
omp_wait_cond(&cond,&lock);
printf("Thr=%d\n",tid);fflush(stdout);
}
} else
{
#pragma omp barrier
printf("Th=%d\n",tid);fflush(stdout);
}
}
}