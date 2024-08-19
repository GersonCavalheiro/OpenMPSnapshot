#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main( int argc,char *argv[])
{
int t;				
int N;				
int **A;
int **B;
int num_of_lines; 		
int m=0;			
int min;			
int min_row;
int min_column;
int flag=1;			
int row_sum=0;			
printf("Type the number of threads you'd like to use: ");
scanf("%d", &t);
omp_set_num_threads(t);
printf("Type the main dimension (N) for your \"A\" input matrix: ");
scanf("%d", &N);
A = (int **)malloc(N*sizeof(int *));
for (int i=0; i<N; i++)
A[i]=(int *)malloc(N*sizeof(int));
B = (int **)malloc(N*sizeof(int *));
for (int i=0; i<N; i++)
B[i]=(int *)malloc(N*sizeof(int));
printf("\n");
for(int i=0;i<N;i++)
{
for(int j=0;j<N;j++)
{
printf("A[%d][%d]= ", i, j);
scanf("%d", &A[i][j]);
}
printf("\n");
}
#pragma omp parallel shared(A, N, num_of_lines, flag)
{
num_of_lines=N/omp_get_num_threads();
#pragma omp for schedule(static, num_of_lines)
for(int i=0;i<N;i++)
{
int row_sum=0;
for(int j=0;j<N;j++)
{
if(i!=j)
row_sum+=abs(A[i][j]);
}
if(abs(A[i][i])<=row_sum)
flag=0;
}
}
if(!flag)
{
printf("\"A\" matrix is not strictly diagonally dominant\n");
}
else
{
printf("\n********************************************************\n");
printf("********************************************************\n");
printf("The \"A\" %dx%d matrix:\n", N, N);
for(int i=0;i<N;i++)
{
for(int j=0;j<N;j++)
{
printf("%d \t", A[i][j]);
}
printf("\n");
}
printf("\n");
#pragma omp parallel shared(A, N, num_of_lines, m)
{
num_of_lines=N/omp_get_num_threads();
#pragma omp for schedule(static, num_of_lines) reduction(max:m)
for(int i=0;i<N;i++)
for(int j=0;j<N;j++)
if(i==j)
if(abs(A[i][j])>m)
m=abs(A[i][j]);
}
printf("Total (absolute) maximum diagonal element of \"A\" is %d\n", m);
printf("********************************************************\n");
#pragma omp parallel shared(A, B, N, num_of_lines, m)
{
#pragma omp for schedule(static, num_of_lines) collapse(2)
for(int i=0;i<N;i++)
for(int j=0;j<N;j++)
if(i==j)
B[i][j]=m;
else
B[i][j]=m-abs(A[i][j]);
}
printf("The \"B\" %dx%d matrix:\n", N, N);
for(int i=0;i<N;i++)
{
for(int j=0;j<N;j++)
{
printf("%d \t", B[i][j]);
}
printf("\n");
}
printf("\n");
#pragma omp parallel shared(B, N, num_of_lines)
{
min=9999;	
min_row=0;
min_column=0;
#pragma omp for schedule(static, num_of_lines) collapse(2)
for(int i=0;i<N;i++)
for(int j=0;j<N;j++)
if(B[i][j]<min)
{
#pragma omp critical (inc_min)
{
min=B[i][j];
min_row=i;
min_column=j;
}
}
}
printf("Minimum element of \"B\" is %d at %dx%d\n", min, min_row, min_column);
printf("********************************************************\n");
printf("********************************************************\n");
}
return(0);
}
