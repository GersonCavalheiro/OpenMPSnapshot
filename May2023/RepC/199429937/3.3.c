#include <omp.h>
#include <stdio.h>
int main() {
int i;
int a[100000];
int n=100000;
for(i=0;i<100000;i++)
a[i]=i;
double b[100000],d[10];
double start_time = omp_get_wtime();
#pragma omp parallel
{
#pragma omp for schedule(static,1000)
for(i=0; i<n-1;i++)
{
b[i]=(a[i]+a[i+1])/2.0;
}
}
printf("Total time = %lf seconds\n",omp_get_wtime()-start_time);
return 0;
}
