#include<stdio.h>
#include<omp.h>
#define MAX 100000000
int array[MAX];
int main(){
int sum=0;
double t1,t2;
for(int i=0; i<MAX; i++){
array[i]=1;
}
t1=omp_get_wtime();
int tid, numt,i;
#pragma omp parallel default(shared) private(i,tid)
{
int partial_sum=0;
tid=omp_get_thread_num();
numt=omp_get_num_threads();
printf("Thread ID %d is working on the array\n",tid);
#pragma omp for
for(i=0; i<MAX; i++){
partial_sum+=array[i];
}
#pragma omp critical
sum+=partial_sum;
}
t2=omp_get_wtime();
printf("Sum is %d\nTime taken is %g\n",sum,t2-t1);
return 0;
}