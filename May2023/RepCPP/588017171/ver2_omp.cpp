#include <bits/stdc++.h>
#include <omp.h>

using namespace std;

int stride = 50;

void SUMMA(double** A, double** B, double** C, int n, int num_threads_our)
{
for(int kk=0; kk<n; kk+=stride)
{
#pragma omp parallel for schedule(static) num_threads (num_threads_our) 
for(int ii=0; ii<n; ii+=stride)
{
for(int jj=0; jj<n; jj+=stride)
{
for(int k=kk; k<kk+stride; ++k) 
{
for(int i=ii; i<ii+stride; ++i)
{
for(int j=jj; j<jj+stride; ++j)
{
C[i][j] += A[i][k]*B[k][j];
}
} 
}              
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
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
{
matrix[i][j] = ((double)rand() / (double)RAND_MAX)*((double)RAND_MAX - 1);
}
}
}

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

int Validate(double** A, double** B, int n) 
{
int mistakes = 0;
for (int i = 0; i < n; i++) 
{
for (int j = 0; j < n; j++) 
{
if (A[i][j] != B[i][j]) mistakes++;
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
int threads = atoi(argv[2]);
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
C[i][j] = 0;
}
}

for (int i = 0; i < n; i++) 
{
for (int j = 0; j < n; j++) 
{
D[i][j] = 0;
}
}

auto now = chrono::system_clock::now();
SUMMA(A, B, C, n, threads);
cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

now = chrono::system_clock::now();
Serial(A, B, D, n);
cout << "millisec=" << std::chrono::duration_cast<std::chrono::milliseconds>(chrono::system_clock::now() - now).count() << "\n";

cout << Validate(C, D, n) << endl;

return 0;
}