#include "stdafx.h"
#include <iostream>
#include <omp.h>

#define NTHREADS 4
#define NRA 1500
#define NCA 1500


using namespace std;
double ** create_2d_matrix(int columns, int rows)
{
double ** mat = new double *[rows];
for (int i = 0; i<rows; i++)
{
mat[i] = new double[columns];
}

return mat;
}
int main()
{
double average = 0;
int i, j;
double time1 = omp_get_wtime();
int NRA_thr = NRA / NTHREADS;
int Nelements = NRA*NCA;

#pragma omp parallel num_threads(NTHREADS) private(i,j)
{
double **a = create_2d_matrix(NCA, NRA_thr);
int id = omp_get_thread_num();

for (i = 0; i<NRA_thr; i++)
for (j = 0; j<NCA; j++)
a[i][j] = 1;

int i_Shift = NRA_thr * id;
for (i = 0; i<NRA_thr; i++)
{
for (j = 0; j<NCA; j++)
{
a[i][j] = (i + i_Shift + j) % 13;
}
}

double sum = 0;
for (i = 0; i<NRA_thr; i++)
for (j = 0; j<NCA; j++)
sum += a[i][j];

#pragma omp critical
{
average += sum / Nelements;
}

}
cout << "<<<PARALLEL>>>" << endl;
cout << "average = " << average << endl;
cout << "Time= " << omp_get_wtime() - time1 << endl;

double time2 = omp_get_wtime();
double **a = create_2d_matrix(NRA, NCA);
double sum = 0;

for (i = 0; i < NRA; i++) {
for (j = 0; j < NCA; j++)
a[i][j] = 1;
}

for (i = 0; i < NRA; i++) {
for (j = 0; j < NCA; j++)
a[i][j] *= (i + j) % 13;
}

for (i = 0; i < NRA; i++) {
for (j = 0; j < NCA; j++)
sum += a[i][j];
}
average = sum / (NRA*NCA);

cout << "<<Sequential code>>" << endl;
cout << "average= " << average << endl;
cout << "Time= " << omp_get_wtime() - time2 << endl;

system("pause");
return 0;
}

