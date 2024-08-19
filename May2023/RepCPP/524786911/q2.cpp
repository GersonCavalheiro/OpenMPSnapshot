#include <iostream>
#include <omp.h>
using namespace std;
#define N 800

int function_call(int j) {
int a;
a = 2 * 2 + j;
return a;
}

int main() {
int i, j;
int a[N][N];
int b[N][N];
int c[N][N];

#pragma omp parallel for private(i, j), schedule(static, 1), collapse(2)
for(i = 1; i <= N; i++)  
for(j = 0; j < N; j++)
b[i - 1][j] = function_call(j);

#pragma omp parallel for private(i, j), schedule(static, 1), collapse(2)
for(j = 0; j < N - 10; j++)
for(i = 0; i < N-10; i++) {
a[i][j + 2] = b[i + 2][j];
c[i + 1][j] = b[i][j + 3];
}
return 0;
}