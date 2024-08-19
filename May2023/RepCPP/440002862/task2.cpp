#include <iostream>
#include <ctime>
#include <omp.h>
#include <cstdlib>
#include <vector>
using namespace std;

int main(int argc, char* argv[])
{
int N = atoi(argv[1]);
vector<vector<int>> A(N), B(N), C(N);
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++)
{ 
A[i].push_back(rand() / 1000);
B[i].push_back(rand() / 1000);
C[i].push_back(0);
}

double start_time = 0;
double total_time = 0;
double end_time = 0;
int eff = 0;
for (int threads = 1; threads < 11; threads++)
{
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads)
{
#pragma omp for collapse(3)
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++)
for (int k = 0; k < N; k++)
C[i][j] += A[i][k] * B[k][j];
}
end_time = omp_get_wtime();
total_time = end_time - start_time;
if (threads == 1)
eff = total_time;

cout << "IJK order, # threads: " << threads << " " << total_time << " " << eff / total_time << endl;
}
cout << endl;

for (int threads = 1; threads < 11; threads++)
{
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads)
{
#pragma omp for collapse(3)
for (int i = 0; i < N; i++)
for (int k = 0; k < N; k++)
for (int j = 0; j < N; j++)
C[i][j] += A[i][k] * B[k][j];
}
end_time = omp_get_wtime();
total_time = end_time - start_time;
if (threads == 1)
eff = total_time;

cout << "IKJ order, # threads: " << threads << " " << total_time << " " << eff / total_time << endl;
}
cout << endl;

for (int threads = 1; threads < 11; threads++)
{
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads)
{
#pragma omp for collapse(3)
for (int j = 0; j < N; j++)
for (int i = 0; i < N; i++)
for (int k = 0; k < N; k++)
C[i][j] += A[i][k] * B[k][j];
}
end_time = omp_get_wtime();
total_time = end_time - start_time;
if (threads == 1)
eff = total_time;

cout << "JIK order, # threads: " << threads << " " << total_time << " " << eff / total_time << endl;
}
cout << endl;

for (int threads = 1; threads < 11; threads++)
{
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads)
{
#pragma omp for collapse(3)
for (int j = 0; j < N; j++)
for (int k = 0; k < N; k++)
for (int i = 0; i < N; i++)
C[i][j] += A[i][k] * B[k][j];
}
end_time = omp_get_wtime();
total_time = end_time - start_time;
if (threads == 1)
eff = total_time;

cout << "JKI order, # threads: " << threads << " " << total_time << " " << eff / total_time << endl;
}
cout << endl;

for (int threads = 1; threads < 11; threads++)
{
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads)
{
#pragma omp for collapse(3)
for (int k = 0; k < N; k++)
for (int j = 0; j < N; j++)
for (int i = 0; i < N; i++)
C[i][j] += A[i][k] * B[k][j];

}
end_time = omp_get_wtime();
total_time = end_time - start_time;
if (threads == 1)
eff = total_time;

cout << "KJI order, # threads: " << threads << " " << total_time << " " << eff / total_time << endl;
}
cout << endl;

for (int threads = 1; threads < 11; threads++)
{
start_time = omp_get_wtime();
#pragma omp parallel num_threads(threads)
{
#pragma omp for collapse(3)
for (int k = 0; k < N; k++)
for (int i = 0; i < N; i++)
for (int j = 0; j < N; j++)
C[i][j] += A[i][k] * B[k][j];
}
end_time = omp_get_wtime();
total_time = end_time - start_time;
if (threads == 1)
eff = total_time;

cout << "KIJ order, # threads: " << threads << " " << total_time << " " << eff / total_time << endl;
}
}