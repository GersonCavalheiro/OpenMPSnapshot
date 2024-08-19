#include <stdlib.h>
#include <stdio.h>
double b[1000][1000];
int main(int argc, char* argv[]) 
{
int i,j;
int n=1000, m=1000;
for (i=0;i<n;i++)
#pragma omp parallel for
for (j=1;j<m;j++)
b[i][j]=b[i][j-1];
printf("b[500][500]=%f\n", b[500][500]);
return 0;
}
