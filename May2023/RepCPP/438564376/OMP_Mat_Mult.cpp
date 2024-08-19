#include <iostream>
#include <time.h>
#include <stdlib.h>
#include <omp.h>
using namespace std;
double** malloc_array(int n)
{
double** array = new double*[n];
for (int i = 0; i < n; i++)
array[i] = new double[n];
return array;
}
void delete_array(double** array, int n)
{
for (int i = 0; i < n; i++)
delete[] array[i];
delete[] array;
}
void generate_array(double** array, int n)
{
srand(time(NULL));
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
array[i][j] = double(rand()) / RAND_MAX;
}
void generate_array_zero(double** array, int n)
{
srand(time(NULL));
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
array[i][j] = 0.0;
}
void print_array(double** array, int n)
{
for(int i = 0; i < n; i++)
{
for(int j = 0; j < n; j++)
cout << array[i][j] << " ";
cout << endl;
}
}

int main(int argc, char** argv)
{
int i, j, k;
const int N = atoi(argv[1]);
double** A, **B, **C;
A = malloc_array(N); generate_array(A, N);
B = malloc_array(N); generate_array(B, N);
C = malloc_array(N); generate_array_zero(C, N);	
double t_start;
cout << "ijk multiplication" << endl;
t_start = omp_get_wtime();
for ( i = 0; i < N; i++)
for ( j = 0; j < N; j++)
for ( k = 0; k < N; k++)
C[i][j] += A[i][k] * B[k][j];
double t_ijk_1stream = omp_get_wtime() - t_start;	
for(int num_threads = 2; num_threads <= 10; num_threads++)
{
generate_array_zero(C, N);
t_start = omp_get_wtime();
#pragma omp parallel for num_threads(num_threads) shared(A, B, C) private(i, j, k)
for (i = 0; i < N; i++)
for (j = 0; j < N; j++)
for (k = 0; k < N; k++)
C[i][j] += A[i][k] * B[k][j];
double t_ijk_parallel = omp_get_wtime() - t_start;
cout << "Multiplication time with " << num_threads << " threads: " << t_ijk_parallel << " seconds, efficiency: " << t_ijk_1stream / t_ijk_parallel << endl;
}
cout << "*********" <<  endl;
cout << "jki multiplication" << endl;
generate_array_zero(C, N);
t_start = omp_get_wtime();
for ( j = 0; j < N; j++)
for ( k = 0; k < N; k++)
for ( i = 0; i < N; i++)
C[i][j] += A[i][k] * B[k][j];
double t_jki_1stream = omp_get_wtime() - t_start;

for(int num_threads = 2; num_threads <= 10; num_threads++)
{
generate_array_zero(C, N);
t_start = omp_get_wtime();
#pragma omp parallel for num_threads(num_threads) shared(A, B, C) private(i, j, k)
for (j = 0; j < N; j++)
for (k = 0; k < N; k++)
for (i = 0; i < N; i++)
C[i][j] += A[i][k] * B[k][j];
double t_jki_parallel = omp_get_wtime() - t_start;
cout << "Multiplication time with " << num_threads << " threads: " << t_jki_parallel << " seconds, efficiency: " << t_jki_1stream / t_jki_parallel << endl;
}
cout << "***********" <<  endl;

cout << "kji multiplication" << endl;
generate_array_zero(C, N);
t_start = omp_get_wtime();
for ( k = 0; k < N; k++)
for ( j = 0; j < N; j++)
for ( i = 0; i < N; i++)
C[i][j] += A[i][k] * B[k][j];
double t_kji_1stream = omp_get_wtime() - t_start;

for(int num_threads = 2; num_threads <= 10; num_threads++)
{
generate_array_zero(C, N);
t_start = omp_get_wtime();
#pragma omp parallel for num_threads(num_threads) shared(A, B, C) private(i, j, k)
for (i = 0; i < N; i++)
for (k = 0; k < N; k++)
for (j = 0; j < N; j++)
C[i][j] += A[i][k] * B[k][j];
double t_kji_parallel = omp_get_wtime() - t_start;
cout << "Multiplication time with " << num_threads << " threads: " << t_kji_parallel << " seconds, efficiency: " << t_kji_1stream / t_kji_parallel << endl;
}
cout << "********" <<  endl;

delete_array(A, N);
delete_array(B, N);
delete_array(C, N);
return 0;
}