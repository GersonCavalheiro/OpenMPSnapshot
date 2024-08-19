

#include <iostream>
#include <random>
#include <omp.h>
using namespace std;

bool debug = false;


template <typename T>
void mat_mult(T A[], T B[], T C[], int m, int n, int k, int thread_count)
{
T *B_trans = new T[n * k];
#pragma omp parallel for num_threads(thread_count)
for (int i = 0; i < n; i++)
#pragma omp parallel for num_threads(thread_count)
for (int j = 0; j < k; j++)
B_trans[j * n + i] = B[i * k + j];

#pragma omp parallel for num_threads(thread_count)
for (int i = 0; i < m; i++)
#pragma omp parallel for num_threads(thread_count)
for (int j = 0; j < k; j++)
{
T sum{};
for (int l = 0; l < n; l++)
sum += A[i * n + l] * B_trans[j * n + l];
C[i * k + j] = sum;
}
delete[] B_trans;
}


double *generate_matrix(int m, int n)
{
default_random_engine generator;
uniform_real_distribution<double> distribution{ 0.0, 1.0 };

double *A = new double[m * n];
for (int i = 0; i < m * n; i++)
A[i] = distribution(generator);
return A;
}


double *read_matrix(int m, int n)
{
double *A = new double[m * n];
for (int i = 0; i < m * n; i++)
cin >> A[i];
return A;
}


void print_matrix(double *A, int m, int n)
{
for (int i = 0; i < m; i++)
{
for (int j = 0; j < n; j++)
cout << A[i * n + j] << " ";
cout << endl;
}
}

int main(int argc, char *argv[])
{
int thread_count = stoi(argv[1]), m = stoi(argv[2]),
n = stoi(argv[3]), k = stoi(argv[4]);

double *A = nullptr, *B = nullptr;
if (debug)
{
cout << "Enter matrix A: " << endl;
A = read_matrix(m, n);
cout << "Enter matrix B: " << endl;
B = read_matrix(n, k);
}
else
{
cout << "Generated matrix A of size " << m << " * " << n << ", "
<< "matrix B of size " << n << " * " << k << endl;
A = generate_matrix(m, n);
B = generate_matrix(n, k);
}

double *C = new double[m * k];
double start = omp_get_wtime();
mat_mult(A, B, C, m, n, k, thread_count);
double finish = omp_get_wtime(), elapsed = finish - start;
cout << "Product calculated. Elapsed time: " << elapsed << " seconds" << endl;

if (debug)
{
cout << "The product is: " << endl;
print_matrix(C, m, k);
}

delete[] A;
delete[] B;
delete[] C;
return 0;
}
