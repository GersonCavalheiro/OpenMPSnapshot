

#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stack>
using namespace std;

void GenMat(double *A, int size)
{
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size; j++)
{
A[i * size + j] = rand() % 10 + 1;
}
}
}

void GenVec(double *A, int size)
{
for (int i = 0; i < size; i++)
{
A[i] = rand() % 5;
}
}

void PrintMat(double *A, int size)
{
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size; j++)
{
cout << A[i * size + j] << " ";
}
cout << endl;
}
cout << endl;
}

void PrintVec(double *A, int size)
{
for (int i = 0; i < size; i++)
{
cout << A[i] << endl;
}
cout << endl;
}

int main(int argc, char *argv[])
{
int n, num_threads, ToPrint;
if (argc > 3)
{
n = atoi(argv[1]);
num_threads = atoi(argv[2]);
ToPrint = atoi(argv[3]);
}
else
{
n = 8;
num_threads = 2; 
ToPrint = 1;
}

double *A = new double[n * n];
double *b = new double[n];
double *x = new double[n];
double *check = new double[n * n];
int *calculated = new int[n];
stack<int> elim;
for (int i = 0; i < n; i++)
{
calculated[i] = -1;
}

int comm_size, my_rank;
MPI_Comm comm = MPI_COMM_WORLD;
MPI_Init(&argc, &argv);
MPI_Comm_size(comm, &comm_size);
MPI_Comm_rank(comm, &my_rank);
int block = n / comm_size;
double *myA = new double[block * n];
double *myb = new double[block];
double *max_row = new double[n];

if (my_rank == 0)
{
srand(time(NULL));
GenMat(A, n);
GenVec(b, n);

MPI_Bcast(A, n * n, MPI_DOUBLE, 0, comm);
MPI_Bcast(b, n, MPI_DOUBLE, 0, comm);
}
else
{
MPI_Bcast(A, n * n, MPI_DOUBLE, 0, comm);
MPI_Bcast(b, n, MPI_DOUBLE, 0, comm);
}

if (my_rank == 0 && ToPrint)
{
cout << "Matrix A:" << endl;
PrintMat(A, n);
cout << "Vector b:" << endl;
PrintVec(b, n);
for (int i = 0; i < n * n; i++)
{
check[i] = A[i];
}
}

MPI_Barrier(comm);

double start_time = MPI_Wtime();

if (my_rank == 0)
{
for (int j = 0; j < n; j++)
{
double max_coff = 0;
int max_index;
double max_b;

for (int i = 0; i < n; i++)
{
if (calculated[i] == -1 && abs(A[i * n + j]) > abs(max_coff))
{
max_coff = A[i * n + j];
max_index = i;
}
}

max_b = b[max_index];
calculated[max_index] = j;
elim.push(max_index);

for (int i = 0; i < n; i++)
{
max_row[i] = A[max_index * n + i];
}

MPI_Bcast(max_row, n, MPI_DOUBLE, 0, comm);
MPI_Bcast(&max_b, 1, MPI_DOUBLE, 0, comm);
MPI_Bcast(calculated, n, MPI_INT, 0, comm);
MPI_Scatter(A, block * n, MPI_DOUBLE, myA, block * n, MPI_DOUBLE, 0, comm);
MPI_Scatter(b, block, MPI_DOUBLE, myb, block, MPI_DOUBLE, 0, comm);

#pragma omp parallel for num_threads(num_threads) shared(myA, myb) schedule(dynamic)
for (int i = 0; i < block; i++)
{
if (calculated[i] == -1)
{
double tmp_coff = myA[i * n + j] / max_row[j];
myA[i * n + j] = 0;
for (int k = j + 1; k < n; k++)
{
myA[i * n + k] -= tmp_coff * max_row[k];
}
myb[i] -= tmp_coff * max_b;
}
}
MPI_Gather(myA, block * n, MPI_DOUBLE, A, block * n, MPI_DOUBLE, 0, comm);
MPI_Gather(myb, block, MPI_DOUBLE, b, block, MPI_DOUBLE, 0, comm);
}
}
else
{
for (int j = 0; j < n; j++)
{
double max_b;

MPI_Bcast(max_row, n, MPI_DOUBLE, 0, comm);
MPI_Bcast(&max_b, 1, MPI_DOUBLE, 0, comm);
MPI_Bcast(calculated, n, MPI_INT, 0, comm);
MPI_Scatter(A, block * n, MPI_DOUBLE, myA, block * n, MPI_DOUBLE, 0, comm);
MPI_Scatter(b, block, MPI_DOUBLE, myb, block, MPI_DOUBLE, 0, comm);
#pragma omp parallel for num_threads(num_threads) shared(myA, myb) schedule(dynamic)
for (int i = 0; i < block; i++)
{
if (calculated[i + my_rank * block] == -1)
{
double tmp_coff = myA[i * n + j] / max_row[j];
myA[i * n + j] = 0;
for (int k = j + 1; k < n; k++)
{
myA[i * n + k] -= tmp_coff * max_row[k];
}
myb[i] -= tmp_coff * max_b;
}
}

MPI_Gather(myA, block * n, MPI_DOUBLE, A, block * n, MPI_DOUBLE, 0, comm);
MPI_Gather(myb, block, MPI_DOUBLE, b, block, MPI_DOUBLE, 0, comm);
}
}

MPI_Barrier(comm);

if (my_rank == 0)
{
for (int j = n - 1; j >= 0; j--)
{
int cur_index = elim.top();
elim.pop();
calculated[cur_index] = -1;
for (int i = 0; i < n; i++)
{
if (calculated[i] != -1)
{
double tmp_coff = A[i * n + j] / A[cur_index * n + j];
A[i * n + j] = 0;
b[i] -= tmp_coff * b[cur_index];
}
}
b[cur_index] /= A[cur_index * n + j];
A[cur_index * n + j] = 1;
x[j] = b[cur_index];
}

double end_time = MPI_Wtime();

if (ToPrint)
{
cout << "Answer x:" << endl;
PrintVec(x, n);

cout << "Check x:" << endl;
for (int i = 0; i < n; i++)
{
double sum = 0;
for (int j = 0; j < n; j++)
{
sum += check[i * n + j] * x[j];
}
cout << sum << endl;
}
}

cout << "Time: " << end_time - start_time << endl;
}

delete[] A;
delete[] b;
delete[] x;
delete[] check;
delete[] calculated;
delete[] myA;
delete[] myb;
delete[] max_row;

MPI_Finalize();
return 0;
}