#include<stdio.h>
#include<omp.h>
int main()
{
int nthreads,tid;
int i,j;
omp_lock_t lock;
omp_set_num_threads(10);
#pragma omp parallel shared(lock) private(tid)
{
tid=omp_get_thread_num();
if(tid==0)
{
omp_init_lock(&lock);
omp_set_lock(&lock);
#pragma omp barrier
i=1;
printf("Introduceti\n");fflush(stdout);
while(i==1)
{
scanf("%d",&j);fflush(stdin);
if(j==1)
{
omp_unset_lock(&lock);
i=0;
}
}
printf("Th=%d\n",tid);fflush(stdout);
} else
if((tid==2)||(tid==3)||(tid==8))
{
#pragma omp barrier
omp_set_lock(&lock);
printf("Thr=%d\n",tid);fflush(stdout);
omp_unset_lock(&lock);
} else
{
#pragma omp barrier
printf("Th=%d\n",tid);fflush(stdout);
}
}
}