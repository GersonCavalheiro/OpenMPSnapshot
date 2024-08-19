#include <iostream>
#include <stdlib.h>
#include <chrono>
#include <time.h>
#include <omp.h>

#define MXY
#define NXY
using namespace std;

void multiplyMatrices(int firstMatrix[M][N], int secondMatrix[M][N], int mult[M][N], int rowFirst, int columnFirst, int rowSecond, int columnSecond);

static int a[M][N] = {
0
}, b[M][N] = {
0
}, c[M][N] = {
0
};

int main() {

int j;
for (int i = 0; i < M; ++i)
for (j = 0; j < N; ++j) {
a[i][j] = rand() % 10;
b[i][j] = rand() % 10;
}

auto start = chrono::high_resolution_clock::now();
multiplyMatrices(a, b, c, M, N, M, N);
auto finish = chrono::high_resolution_clock::now();
chrono::duration < double > elapsed = finish - start;

cout << (int)(elapsed.count() * 1000000) << endl;

return 0;
}

void multiplyMatrices(int firstMatrix[M][N], int secondMatrix[M][N], int mult[M][N], int rowFirst, int columnFirst, int rowSecond, int columnSecond) {
int i, j, k;

#pragma omp parallel for private(j, k)
for (i = 0; i < rowFirst; ++i)
for (j = 0; j < columnSecond; ++j)
for (k = 0; k < columnFirst; ++k)
mult[i][j] += firstMatrix[i][k] * secondMatrix[k][j];

}
