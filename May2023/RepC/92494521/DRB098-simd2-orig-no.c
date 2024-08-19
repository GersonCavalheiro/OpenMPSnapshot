#include <stdio.h>
int main()
{
int len=100;
double a[len][len], b[len][len], c[len][len];
int i,j;
for (i=0;i<len;i++)
for (j=0;j<len;j++)
{
a[i][j]=((double)i)/2.0; 
b[i][j]=((double)i)/3.0; 
c[i][j]=((double)i)/7.0; 
}
#pragma omp simd collapse(2)
for (i=0;i<len;i++)
for (j=0;j<len;j++)
c[i][j]=a[i][j]*b[i][j];
printf ("c[50][50]=%f\n",c[50][50]);
return 0;
}
