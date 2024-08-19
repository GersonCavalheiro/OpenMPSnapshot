#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#define MAX 1000
void oddEvenSort(int* a,int n,int threads)
{
int phase,i,temp;
#pragma omp parallel default(none) shared(a,n) private(i,temp,phase)
{
for(phase=0;phase<n;phase++)
{
if(phase%2==0)
{
#pragma omp for
for(i=1;i<n;i+=2)
{
if(a[i-1]>a[i])
{
temp=a[i];
a[i]=a[i-1];
a[i-1]=temp;
}
}
}
else
{
#pragma omp for
for(i=1;i<n-1;i+=2)
{
if(a[i]>a[i+1])
{
temp=a[i];
a[i]=a[i+1];
a[i+1]=temp;
}
}
}
}
}
}
int main()
{
int A[MAX];
for(int i=0;i<MAX;i++)
A[i]=rand()%MAX;
omp_set_num_threads(8);
double time=omp_get_wtime();
oddEvenSort(A,MAX,4);
time=omp_get_wtime()-time;
printf("Sorted Array :-\n");
for(int i=0;i<MAX;i++)
{
printf("%d ",A[i]);
}
printf("\n");
printf("The time taken is %f\n",time);
return 0;
} 
