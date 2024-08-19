#include<stdio.h>
#include<omp.h>
int main() {
int n,a[100],i;
omp_set_num_threads(2);
printf("---------------created by Aryan Chandrakar------------------\n\n");
printf("[-] enter the no of terms of fibonacci series which have to be generated: ");
scanf("%d",&n);
a[0]=0;
a[1]=1;
#pragma omp parallel
{
#pragma omp single
for(i=2;i<n;i++)
{
a[i]=a[i-2]+a[i-1];
printf("   [~] thread involved in the computation of fib no %d is=%d\n",i+1,omp_get_thread_num());
}
#pragma omp barrier
#pragma omp single
{
printf("\n[+] the elements of fib series are\n\n");
for(i=0;i<n;i++)
printf("  [~]  %d   -->thread displaying this no is =  %d\n",a[i],omp_get_thread_num());
}
}
return 0;
}
