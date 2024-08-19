#include<stdio.h>
#include<omp.h>
int main()
{
int N= 5;
#pragma omp parallel num_threads(4){ 
#pragma omp for schedule(static,1) 
for(int i= 0; i < N; i++){
#pragma omp ordered
{
printf("%d\n",omp_get_thread_num());
}
}
}
}
