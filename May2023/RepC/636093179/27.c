#include <stdio.h>
#include <omp.h>
int main()
{
int N= 5;
#pragma omp parallel
{
int id= omp_get_thread_num();
#pragma omp for collapse(3)
for(int i= 0; i < N; i++){
for(int j= 0; j < N; j++){
for(int k= 0; k < N; k++){
printf("Thread %d: %d %d %d\n",id,i,j,k);
}
}
}
}
}
