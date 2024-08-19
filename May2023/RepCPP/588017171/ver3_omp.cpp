#include <bits/stdc++.h>
#include <omp.h>
#include "cblas.h"
#include <unistd.h>

using namespace std;

int stride = 40;
int num_threads_our;

void PrintMatrix(double** matrix, int n)
{
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
cout << matrix[i][j] << " ";
}
cout << endl;
}
}

void flatten(double** matrix, double* matrix_prime, int row_start, int row_end, int col_start, int col_end) {
for (int i = row_start; i < row_end; i++)
{
for (int j = col_start; j < col_end; j++)
{
matrix_prime[(i - row_start) * stride + j - col_start] = (double)matrix[i][j];
}
}
}

void reshape(double** A, double* A_prime, int row_start, int row_end, int col_start, int col_end) {
for (int i = row_start; i < row_end; i++)
{
for (int j = col_start; j < col_end; j++)
{
A[i][j] += (double)A_prime[(i - row_start) * stride + j - col_start];
}
}
}

void SUMMA(double** A, double** B, double** C, int n, int num_threads_our)
{

double alpha = 1.0;
double beta = 0.0;
for (int kk = 0; kk < n; kk += stride)
{
#pragma omp parallel for schedule(static) num_threads (num_threads_our) 
for (int ii = 0; ii < n; ii += stride)
{
double* A_prime = (double*)malloc(stride * stride * sizeof(double));
flatten(A, A_prime, ii, ii + stride, kk, kk + stride);

for (int jj = 0; jj < n; jj += stride)
{
double* B_prime = (double*)malloc(stride * stride * sizeof(double));
double* C_prime = (double*)malloc(stride * stride * sizeof(double));

flatten(B, B_prime, kk, kk + stride, jj, jj + stride);
flatten(C, C_prime, ii, ii + stride, jj, jj + stride);

cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, stride, stride, stride, alpha, A_prime, stride, B_prime, stride, beta, C_prime, stride);

reshape(C, C_prime, ii, ii + stride, jj, jj + stride);
}
}
}
}

void Serial(double** A, double** B, double** C, int n)
{
for (int k = 0; k < n; k++)
{
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
C[i][j] += A[i][k] * B[k][j];
}
}
}
}

void Initialize(double** matrix, int n)
{
srand((unsigned int)time(NULL));
sleep(1);
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
matrix[i][j] = ((double)rand() / (double)RAND_MAX) * ((double)RAND_MAX - 1);
}
}
}

int Validate(double** A, double** B, int n) 
{
int mistakes = 0;
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
if ((float)A[i][j] != (float)B[i][j]) {
mistakes++;
}
}
}
return mistakes;
}

int main(int argc, char const* argv[])
{
if (argc != 3) {
cout << "Usage: ./ver0 <matrix_size> <num_threads>\n\nmatrix_size: Positive Integer\nnum_threads: Positive Integer" << endl;
return 1;
}
int n = atoi(argv[1]);
num_threads_our = atoi(argv[2]);
double** A = new double* [n];
double** B = new double* [n];
double** C = new double* [n];
double** D = new double* [n];

for (int i = 0; i < n; i++)
{
A[i] = new double[n];
B[i] = new double[n];
C[i] = new double[n];
D[i] = new double[n];
}

Initialize(A, n);
Initialize(B, n);
for (int i = 0; i < n; i++) 
{
for (int j = 0; j < n; j++) 
{
C[i][j] = 0.0;
}
}

for (int i = 0; i < n; i++) 
{
for (int j = 0; j < n; j++) 
{
D[i][j] = 0.0;
}
}

openblas_set_num_threads(1);

auto now = chrono::system_clock::now();
SUMMA(A, B, C, n, num_threads_our);
cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

now = chrono::system_clock::now();
Serial(A, B, D, n);
cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

cout << Validate(C, D, n) << endl;

return 0;
}