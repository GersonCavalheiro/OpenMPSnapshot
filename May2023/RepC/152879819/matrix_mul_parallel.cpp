#include <iostream>
#include <omp.h>
#define N 2000
using namespace std;
int A[N][N];
int B[N][N];
int C[N][N];
int main() {
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
A[i][j] = 1;
B[i][j] = 1;
}
}
double start_t = omp_get_wtime();
#pragma omp parallel num_threads(4)
{
#pragma omp for
for (int i=0; i<N; i++) {
for (int j=0; j<N; j++) {
int num = 0;
for (int k=0; k < N; k++) {
num += A[i][k] * B[k][j];
}
C[i][j] = num;
}
}
}
double end_t = omp_get_wtime();
cout << end_t - start_t << "\n";
}