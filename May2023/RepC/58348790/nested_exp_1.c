#include <stdio.h>
#include <omp.h>
static float a[10][10], b[10][10];
int main( ) 
{
int i,j,k, sum;
int n =10;
#pragma omp parallel private(sum,i)
{
for(j=0;j<n;j++) { 
for(i=0;i<n;i++) { 
sum = 0; 
#pragma omp parallel 
for(k=0;k<n;k++) { 
sum += a[i][k] * b[k][j]; 
} 
} 
} 
}
return 0;
}
