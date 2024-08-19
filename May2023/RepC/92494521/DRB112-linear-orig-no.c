#if (_OPENMP<201511)
#error "An OpenMP 4.5 compiler is needed to compile this test."
#endif
#include <stdio.h>
int main()
{
int len=100;
double a[len], b[len], c[len];
int i,j=0;
for (i=0;i<len;i++)
{
a[i]=((double)i)/2.0; 
b[i]=((double)i)/3.0; 
c[i]=((double)i)/7.0; 
}
#pragma omp parallel for linear(j)
for (i=0;i<len;i++)
{
c[j]+=a[i]*b[i];
j++;
}
printf ("c[50]=%f\n",c[50]);
return 0;
}
