#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main()
{
int n=5;
int a[n][n],b[n][n],c[n][n];
int q=0,i,j,k,t[n][n];
for (i= 0; i< n; i++)
{ 
for (j= 0; j< n; j++)
{
a[i][j]=q;
b[i][j]=q;
q++;
c[i][j]=0;
}
}
#pragma omp parallel for shared(a, b, c) private(i,j,k)
for (i = 0; i < n; ++i) {
for (j = 0; j < n; ++j) {
for (k = 0; k < n; ++k) {
t[i][j]=omp_get_thread_num();
c[i][j] += a[i][k] * b[k][j];
}
}
}	
printf("[+] Matrix multiplied by Aryan : \n");
for (i= 0; i< n; i++)
{
for (j= 0; j< n; j++)
{
printf(" %d - found by thread %d \t",c[i][j],t[i][j]);
}
printf("\n\n");
}	
}
