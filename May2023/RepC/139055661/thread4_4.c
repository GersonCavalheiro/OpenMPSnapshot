#include<omp.h>
#include<stdio.h>
int main()
{
int prime=1,i,N; 
printf("Enter a number to check whether it is prime or not: ");
scanf("%d",&N);
int num=(N/2) + 1;
#pragma omp parallel private (prime)
prime=1;
#pragma omp for
for (i=2; i<num; i++)
{
if(N%i==0){
prime = 0;
}
}
#pragma omp critical
if(prime==1)
printf("%d is prime\n",N);
else
printf("%d is not prime\n",N);
}
