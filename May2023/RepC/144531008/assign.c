#include<stdio.h>
#include<omp.h>
void main(){
int n,x=0;
printf("Enter the value of n: ");
scanf("%d",&n);
int arr[n];
printf("Enter array elements: ");
for(int i=0;i<n;i++){
scanf("%d",arr+i);
}
#pragma omp parallel num_threads(4) reduction(+:x)
{
#pragma omp for
for(int i=0;i<n;i++)
x+=arr[i];
}
printf("%d\n",x);
}