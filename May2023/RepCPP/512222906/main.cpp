#include<iostream>
#include <math.h>
#include <omp.h>
#include <cstdlib>
#include<cstring>

typedef double TYPE;

#define  MIN(a,b) ((a)<(b)) ? (a) : (b)

const int N = (1 << 11);

const int b = 32;	

const bool calculateErrors = false;

using namespace std;


void init(TYPE*& A, const int m = N, const int n = N)
{
A = new TYPE[m * n];
}

void free(TYPE* A)
{
delete[] A;
}

void fill(TYPE* A)
{
for (int i = 0; i < N; ++i)
for (int j = 0; j < N; ++j)
{
A[i * N + j] = rand() / 1000.0;
}
}

void zero(TYPE* A, int n = N, int m = N)
{
for (int i = 0; i < n; ++i)
for (int j = 0; j < m; ++j)
{
A[i * m + j] = 0.0;
}
}

void show(TYPE const* A, const int rows = N, const int cols = N)
{
for (int i = 0; i < rows; ++i) {
for (int j = 0; j < cols; ++j)
cout << A[i * cols + j] << "\t";
cout << "\n";
}
}


TYPE LUDecompositionParallel(TYPE* A)
{

TYPE t = -omp_get_wtime();

for (int i = 0; i < N - 1; ++i)
{
#pragma omp parallel for default(none) shared(A,i)
for (int j = i + 1; j < N; ++j)
{
A[j * N + i] /= A[i * N + i];
}
#pragma omp parallel for default(none) shared(A,i) schedule(guided,4)
for (int j = i + 1; j < N; ++j)
for (int k = i + 1; k < N; ++k)
A[j * N + k] -= A[j * N + i] * A[i * N + k];

}
t += omp_get_wtime();
return t;
}

void rectangleLUij(TYPE* A, const int m, const int n, const int i0, const int j0) 
{
int min = MIN(m - 1, n);
for (int i = 0; i < min; ++i)
{
for (int j = i + 1; j < m; ++j)
A[(j + i0) * N + i + j0] /= A[(i + i0) * N + i + j0];
if (i < n - 1)
{
for (int j = i + 1; j < m; ++j)
for (int k = i + 1; k < n; ++k)
A[(j + i0) * N + k + j0] -= A[(j + i0) * N + i + j0] * A[(i + i0) * N + k + j0];
}
}
}

void rectangleLUijParallel(TYPE* A, const int m, const int n, const int i0, const int j0) 
{
int min = MIN(m - 1, n);
for (int i = 0; i < min; ++i)
{
#pragma omp parallel for default(none) shared(A,i,min,m,n,i0,j0)
for (int j = i + 1; j < m; ++j)
A[(j + i0) * N + i + j0] /= A[(i + i0) * N + i + j0];
if (i < n - 1)
{
#pragma omp parallel for default(none) shared(A,i,min,m,n,i0,j0) 
for (int j = i + 1; j < m; ++j)
for (int k = i + 1; k < n; ++k)
A[(j + i0) * N + k + j0] -= A[(j + i0) * N + i + j0] * A[(i + i0) * N + k + j0];
}
}
}



void duplicate(TYPE* A, TYPE* Ad, int colsA, int rowsAd, int colsAd, int i0, int j0)
{

for (int i = 0; i < rowsAd; ++i)
{
for (int j = 0; j < colsAd; ++j)
{
Ad[i * colsAd + j] = A[(i0 + i) * colsA + (j0 + j)];
}
}

}


TYPE LUDecompositionBlockParallel(TYPE* A)
{
TYPE t = -omp_get_wtime();
TYPE* U23;
init(U23, b, N - b);
TYPE* L32;
init(L32, N - b, b);
for (int i = 0; i < N - 1; i += b)
{

rectangleLUijParallel(A, N - i, b, i, i);

#pragma omp parallel default(none) shared(A,L32,U23,i)
{
#pragma omp for 
for (int j = 0; j < N - b - i; ++j)
{

for (int k = 0; k < b; ++k)
{
TYPE sum = 0;
for (int q = 0; q < k; ++q) {
sum += A[(i + k) * N + (i + q)] * A[(q + i) * N + b + i + j];
}
A[(k + i) * N + b + i + j] -= sum;
}

}

#pragma omp for
for (int j = 0; j < N - b - i; ++j)
for (int k = 0; k < b; ++k)
{
U23[k * (N - i - b) + j] = A[(i + k) * N + j + i + b];
L32[j * b + k] = A[(i + j + b) * N + k + i];
}


#pragma omp for
for (int j = b + i; j < N; ++j)
{
for (int c = 0; c < b; ++c)
{
for (int k = b + i; k < N; ++k) {
A[j * N + k] -= L32[(j - i - b) * b + c] * U23[c * (N - i - b) + k - (i + b)];
}

}
}
}


}
free(U23); free(L32);

t += omp_get_wtime();
return t;
}


TYPE getError(TYPE* LU, TYPE* A)
{
double t = -omp_get_wtime();
TYPE Amid = A[N / 2];

#pragma omp parallel for default(none) shared(A,LU)
for (int i = 0; i < N; ++i)
for (int j = 0; j < N; ++j)
{
int min;
bool ismallerj = false;
if (i <= j)
{
ismallerj = true;
min = i;
}
else min = j;
for (int k = 0; k < min; ++k)
A[i * N + j] -= LU[i * N + k] * LU[k * N + j];
if (ismallerj) {
A[i * N + j] -= LU[min * N + j];
}
else {
A[i * N + j] -= LU[i * N + min] * LU[min * N + j];
}
}

TYPE max_a = 0;
TYPE a = 0;

#pragma omp parallel for default(none) shared(A,max_a) private(a)
for (int i = 0; i < N; ++i)
for (int j = 0; j < N; ++j)
{
if ((a = fabs(A[i * N + j])) > max_a) {
#pragma omp critical
if (a > max_a) {
max_a = a;
}
}

}


t += omp_get_wtime();
cout << "\nerror_time = " << t << "\n";

return max_a / Amid;
}



int main()
{
setlocale(LC_ALL, "RUS");
cout << "N = " << N << endl;
cout << "b = " << b << endl;

TYPE* A, * Ad, * LU;	
init(A); init(Ad); init(LU);
fill(A);
memcpy(Ad, A, N * N * sizeof(TYPE));
memcpy(LU, A, N * N * sizeof(TYPE));
TYPE er; TYPE tm;



cout << "\n\t\t LU- " << endl;
tm = LUDecompositionParallel(LU);	
cout << ": " << tm << endl;
if (calculateErrors) {
er = getError(LU, Ad);
cout << " ||LU-A||: " << er << endl;
memcpy(Ad, A, N * N * sizeof(TYPE));
}
memcpy(LU, A, N * N * sizeof(TYPE));




cout << "\n\t\t LU- " << endl;
tm = LUDecompositionBlockParallel(LU);	
cout << ": " << tm << endl;
if (calculateErrors) {
er = getError(LU, Ad);
cout << " ||LU-A||: " << er << endl;
memcpy(Ad, A, N * N * sizeof(TYPE));
}



free(A); free(Ad); free(LU);

cin.get();
return 0;