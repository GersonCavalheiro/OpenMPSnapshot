#include<omp.h>
#include<stdio.h>
int main(){
int sum = 0;
int array[100];
for(int i = 0; i < 100; i++){
array[i] = i;
}    
#pragma omp parallel
{
for(int i = 0; i < 100; i++){
sum += array[i];
printf("%d=%d\n",omp_get_thread_num(),sum);
}
printf("\n");
}
printf("Sum is %d",sum);
return 0;
}
