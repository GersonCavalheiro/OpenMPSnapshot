#include <omp.h>
#include<stdlib.h>
#include<stdio.h>
void work() {
omp_lock_t lock;
omp_init_lock(&lock);
#pragma omp parallel
{
int i;
#pragma omp for
for (i = 0; i < 100; i++) {
#pragma omp task 
{ 
omp_set_lock(&lock);
#pragma omp task
{  }
omp_unset_lock(&lock);
}
}
}
omp_destroy_lock(&lock);
}
int main()
{
work();
return 0;
}
