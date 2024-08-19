#include <stdio.h>
#include <omp.h>
int main()
{
int N= 5, i= -1, j= -1;
#pragma omp parallel
{
int j= -1;
int id= omp_get_thread_num();
#pragma omp for
for(i= 0; i < N; i++){
for(j= 0; j < N; j++){
printf("Thread %d: %d %d\n",id,i,j);
}
}
}
}
