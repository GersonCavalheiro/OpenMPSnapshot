#include<omp.h>
#include<stdio.h>
int main()
{
int sum=0;
int lsum=0;
int array[100];
for(int i = 0; i < 100; i++){
array[i] = i;
}
#pragma omp parallel private(lsum)
{
lsum=0;
#pragma omp for
for(int i = 0; i < 100; i++){
lsum += array[i];
}
#pragma omp critical
{	
sum+=lsum;
}
}
printf("The sum is: %d.\n",sum);
return 0;
}
