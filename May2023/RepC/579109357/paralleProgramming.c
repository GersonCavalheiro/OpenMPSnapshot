#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <pthread.h>
#include<omp.h>
int main()
{
int i,n;
printf("Enter value of n : ");
scanf("%d",&n);
clock_t start,end;
float execution_time;
start = clock();
for(i=0;i<n;i++){
}
end = clock();
execution_time = ((double)(end-start))/CLOCKS_PER_SEC;
printf("Execution time is %f\n", execution_time);
start = clock();
#pragma omp parallel for
for(i=0;i<n;i++){
}
end = clock();
execution_time = ((double)(end-start))/CLOCKS_PER_SEC;
printf("Execution time is %f\n", execution_time);
return 0;
}
