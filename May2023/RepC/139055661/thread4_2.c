#include<omp.h>
#include<stdio.h>
int main()
{
int sum=0;
int array[100];
for(int i = 0; i < 100; i++){
array[i] = i;
}
#pragma omp parallel for reduction (+:sum)
for(int i = 0; i < 100; i++){
sum += array[i];
}
printf("The sum is: %d.\n",sum);
return 0;
}
