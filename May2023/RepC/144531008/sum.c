#include <stdio.h>
#include <omp.h>
int rowsum(int a[][100],int r,int c);
int main(int argc, char const *argv[])
{
int n,r;
printf("Enter the value of n:");
scanf("%d",&n);
int row[n];
int cols[n];
int a[n][n];
for(int i=0;i<n;i++){
row[i]=0;
cols[i]=0;
}
printf("Enter the matrix values:\n");
for(int i=0;i<n;i++){
for(int j=0;j<n;j++){
scanf("%d",&a[i][j]);
}
}
#pragma omp for collapse(2)
for(int i=0;i<n;i++){
for(int j=0;j<n;j++){
row[i]+=a[i][j];
cols[i]+=a[j][i];
}
}
for(int i=0;i<n;i++){
printf("%d ",row[i] );
}
printf("\n");
for (int i = 0; i < n; ++i)
{
printf("%d ",cols[i] );
}
printf("\n");
return 0;
}
