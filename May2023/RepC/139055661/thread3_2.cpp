#include<stdio.h>
#include<omp.h>
#include<iostream>
int main(){
int sumOut = 0;
int array[100];
for(int i = 0; i < 100; i++){
array[i] = i;
}
#pragma omp parallel
{
int sum = 0;
for(int i = 0; i < 100; i++){
sum += array[i];
}
printf("%d",sum);
sumOut = sum;
}
printf("Sum is %d",sumOut);
return 0;
}
