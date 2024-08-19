#include <iostream>
#include <omp.h>
using namespace std;
#define N 1024

int main()
{
int i, j;
int X[N][N];
int Y[N];
int Z[N];
int k = 1;

for(i = 0; i < N; i++){                      
Y[i] = k;  
k = k * 2;
Z[i] = -1;
#pragma omp parallel for schedule(static, 1)
for(j = 0; j < N; j++)
X[i][j] = 2;
}

#pragma omp parallel for private(i, j), schedule(static, 1), collapse(2)
for(i = 0; i < N; i++)   
for(j = 0; j < N; j++) {
Z[i] += Y[j] + X[i][j];
}
return 0;
}