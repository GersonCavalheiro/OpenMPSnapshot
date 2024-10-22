#include <stdio.h>
#include <omp.h>
main ()  {
int   i, n, chunk;
float a[100], b[100], result;
n = 100;
chunk = 10;
result = 0.0;
for (i=0; i < n; i++)
{
a[i] = i * 1.0;
b[i] = i * 2.0;
}
#pragma omp parallel for  default(none) shared(a,b,n) private(i) schedule(static)   reduction(+:result)  
for (i=0; i < n; i++)
result = result + (a[i] * b[i]);
printf("Final result= %f\n",result);
}
