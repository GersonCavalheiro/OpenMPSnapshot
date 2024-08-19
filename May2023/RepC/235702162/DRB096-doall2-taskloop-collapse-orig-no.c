#include <stdio.h>
int a[100][100];
int main()
{
int i, j;
{
#pragma omp parallel for private(j ) 
for (i = 0; i < 100; i++)
#pragma omp parallel for private(j ) 
for (j = 0; j < 100; j++)
a[i][j]+=1; 
}
printf ("a[50][50]=%d\n", a[50][50]);
return 0;
}
