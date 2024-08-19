#include <stdio.h>
#include <assert.h>
int main()
{
int i,error;
int len = 1000;
int a[len], b=5;
for (i=0; i<len; i++)
a[i]= i;
#pragma omp parallel shared(b, error) 
{
#pragma omp for nowait
for(i = 0; i < len; i++)
a[i] = b + a[i]*5;
#pragma omp barrier
#pragma omp single
error = a[9] + 1;
}
assert (error == 51);
printf ("error = %d\n", error);
return 0;
}  
