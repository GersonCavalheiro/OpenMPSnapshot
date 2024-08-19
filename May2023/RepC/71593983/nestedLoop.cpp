#include<iostream>
#include<omp.h>
using namespace std;
int main()
{
int maxiter = 1;
int n = 3;
int m = 3;
int j;
double x,y;	
double A[n][m];
#pragma omp parallel for
for(int i = 0; i < n; i++)
{
for(j = 0; j < m; j++)
{
x = i/maxiter;
y = j/maxiter;
A[i][j] = x + m*y;
}
}
for(int i = 0; i < n; i++)
{
for(int j = 0; j < m; j++)
{
cout << A[i][j] << " ";
}
cout << endl;
}
return 0;
}
