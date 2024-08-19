#include <stdio.h>
#include <omp.h>
int main()
{
int N= 4;
omp_set_nested(1);
#pragma omp parallel
{
int id= omp_get_thread_num();
#pragma omp for
for(int i= 0; i < N; i++){
#pragma omp parallel for
for(int j= 0; j < N; j++){
printf("Thread %d %d: %d %d\n",id,omp_get_thread_num(),i,j);
}
}
}
}
