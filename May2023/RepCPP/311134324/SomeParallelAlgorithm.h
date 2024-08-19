#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <omp.h>
#include "../Algorithm.h"

#ifndef SOME_PARALLEL_ALGORITHM
#define SOME_PARALLEL_ALGORITHM

using namespace std;
class SomeParallelAlgorithm : public Algorithm
{
private:
static const int n = 100;
static const int m = 100;

public:
virtual void UpdateParam(int n) {}
virtual void SetThreads(int n) {}
virtual string GetLabel() { return "some_parallel"; }

void ProcessA(double a[n][m])
{
#pragma omp parallel for
for (int i_ = 0; i_ < n * m; i_++)
{
int i = i_ / m;
int j = i_ % m;
a[i][j] = sin(0.00001 * 0.);
}
}

void ProcessB(double array[n][m])
{
#pragma omp parallel for
for (int i_ = 0; i_ < n * m; i_++)
{
int i = i_ / m;
int j = i_ % m;
array[i][j] = 10 * i + j;
}
}

virtual void Run()
{
double a[n][m];
double b[n][m];
int i, j;
#pragma omp task
ProcessA(a);
#pragma omp task
ProcessB(b);
#pragma omp taskwait

#pragma omp parallel for
for (int i_ = 0; i_ < (n - 4) * m; i_++)
{
int i = i_ / m;
int j = i_ % m;
b[i][j] = a[i + 4][j] * 1.5;
}

FILE *ff;
ff = fopen("result.txt", "w");
for (i = 0; i < n; i++)
{
for (j = 0; j < m; j++)
{
fprintf(ff, "%f ", b[i][j]);
}
fprintf(ff, "\n");
}
fclose(ff);
}
};
#endif