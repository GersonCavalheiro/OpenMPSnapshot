#include <stdio.h>
int main(int argc, char* argv[]) 
{
int i,j;
int n=100, m=100;
double b[n][m];
#pragma omp parallel for private(j)
for (i=1;i<n;i++)
for (j=0;j<m;j++) 
b[i][j]=b[i][j-1];
printf ("b[50][50]=%f\n",b[50][50]);
return 0;     
}
