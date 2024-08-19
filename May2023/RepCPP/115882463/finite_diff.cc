#include <iostream>
#include <vector>
#include <utility>
#include <cmath>
#include <math.h>
#include <omp.h>
using namespace std;

typedef vector<double> Row;
typedef vector<Row> Matrix;

int main()
{
double start = omp_get_wtime();
int m = 101;
double e = 10;

Matrix Unew(m, Row(m)),vA(m, Row(m));
Matrix Uold(m, Row(m)),vB(m, Row(m));

#pragma omp parallel num_threads(4) shared(Unew, Uold)
{
#pragma omp for
for (int x=0; x<m; x++)
{
int i = omp_get_thread_num();
printf("Hello from thread %d\n", i);
for (int y=0; y<m; y++)
{
Unew[x][y] = 0;
Uold[x][y] = 0;
}
Unew[x][m-1] = 100;
Uold[x][m-1] = 100;
}
while (e > 0.001)
{
#pragma omp for
for (int x=1; x<(m-1); x++)
{
for (int y=1; y<(m-1); y++)
{
Unew[x][y] = 0.25*(Uold[x+1][y]+Uold[x-1][y]+Uold[x][y+1]+Uold[x][y-1]);
}
}

#pragma omp single
e = 0;

#pragma omp for reduction (+:e)
for (int x=1; x<(m-1); x++)
{
for (int y=1; y<(m-1); y++)
{
e = e + abs(Unew[x][y] - Uold[x][y]);
}
}
#pragma omp for
for (int x=1; x<(m-1); x++)
{
for (int y=1; y<(m-1); y++)
{
Uold[x][y] = Unew[x][y];
}
}
}
}
double end = omp_get_wtime();
cout<< "Time difference is " << end - start << endl;
return 0;
}