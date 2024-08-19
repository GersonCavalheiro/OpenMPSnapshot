#include <stdio.h>
#include <stdlib.h>
int main(int argc, char* argv[]) 
{
int i,j;
int n=100, m=100;
double b[n][m];
#pragma omp parallel for
for (i=1;i<n;i++)
#pragma omp parallel for simd
for (j=0;j<m;j++)
b[i][j]= i * j;
for (i=1;i<n;i++)
#pragma omp parallel for simd
for (j=0;j<m;j++)
b[i][j]=b[i-1][j];
printf ("b[50][50]=%f\n",b[50][50]);
return 0;     
}
