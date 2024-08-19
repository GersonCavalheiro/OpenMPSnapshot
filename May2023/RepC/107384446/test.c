#include<stdio.h>
#include<omp.h>
int main(){
#pragma omp parallel num_threads(5)
{
int tid;
tid=omp_get_thread_num();
printf("%d\n",tid );
}
}
