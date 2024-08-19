#include <stdio.h>
#include <omp.h>
int main()
{
int N= 5;
#pragma omp parallel
{
int id= omp_get_thread_num();
#pragma omp for
for(int i= 0; i < N; i++){
for(int j= 0; j < N; j++){
printf("Thread %d: %d %d\n",id,i,j);
}
}
}
}
