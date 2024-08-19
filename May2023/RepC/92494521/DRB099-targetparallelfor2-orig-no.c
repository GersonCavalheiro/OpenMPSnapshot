#include <stdio.h>
void foo (double* a, double* b, int N)
{
int i; 
#pragma omp target map(to:a[0:N]) map(from:b[0:N])
#pragma omp parallel for
for (i=0;i< N ;i++)
b[i]=a[i]*(double)i;
}
int main(int argc, char* argv[])
{
int i;
int len = 1000;
double a[len], b[len];
for (i=0; i<len; i++)
{
a[i]= ((double)i)/2.0;
b[i]=0.0;
}
foo(a, b, len);
printf("b[50]=%f\n",b[50]);
return 0;
}
