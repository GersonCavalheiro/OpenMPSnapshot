#include<parallel/cond-omp.h>
void omp_wait_cond(omp_cond_t *cond,omp_lock_t *mutex)
{
int test=0;
while(test==0)
{
omp_set_lock(mutex);
if(cond->cond==1)
{
test=1;
if(cond->broadcast==1)
{
cond->counter--;
if(cond->counter==0)
{
cond->cond=0;
cond->broadcast=0;
}
}
}
omp_unset_lock(mutex);
}
}
void omp_signal_cond(omp_cond_t *cond,omp_lock_t *mutex)
{
cond->broadcast=0;
#pragma omp atomic
cond->cond++;
}
void omp_bcast_cond(omp_cond_t *cond,omp_lock_t *mutex,int nr)
{
int test;
test=0;
while(test==0)
{
if(cond->counter==0)
{
if(omp_test_lock(mutex)!=0)
{
omp_unset_lock(mutex);
perror("vous n'avez pas aquire le lock");
}
else
{
cond->broadcast=1;
cond->cond=1;
cond->counter=nr-1;
test=1;
}
}
}
}
void omp_init_cond(omp_cond_t *cond,omp_lock_t *mutex)
{
cond->cond=0;
cond->counter=0;
cond->broadcast=0;
omp_init_lock(mutex);
}
