#include<stdio.h>
#include<omp.h>
int addem(int C[][3], int A[][3], int B[][3], int N);
int makeA(int A[][3],int N);
int makeB(int A[][3],int N);
int print(int A[][3], int N);
int main(void)
{
int N = 3;
int i,j;
int A[N][N];
int B[N][N];
int C[N][N];
#pragma omp parallel shared(C,N)
{ 
#pragma omp sections
{
#pragma omp section
makeA(A,N);
#pragma omp section
makeB(B,N);
}	
addem(C,A,B,N);
}
print(A,N);
printf("\n");
print(B,N);
printf("\n");
print(C,N);
return 0;
}
int addem(int C[][3], int A[][3], int B[][3], int N)
{
int i,j;
#pragma omp for private(j)
for(i = 0; i < N; i++)
{
for(j = 0; j < N; j++)
{
C[i][j] = A[i][j] + B[i][j];
}
}
return 0;
}
int makeB(int A[][3],int N)
{
int i,j;
for(i = 0; i < N; i++)
{
for(j = 0; j < N; j++)
{	
A[i][j] = i*N+j;
}
}
return 0;
}
int makeA(int A[][3],int N)
{
int i,j;
for(i = 0; i < N; i++)
{
for(j = 0; j < N; j++)
{	
A[i][j] = 1;
}
}
return 0;
}
int print(int A[][3], int N)
{
int i,j;
for(i = 0; i < N; i++)
{
for(j = 0; j < N; j++)
{	
printf("%i ", A[i][j]);
}
printf("\n");
}
return 0;
}