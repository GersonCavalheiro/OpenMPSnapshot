#include "GaussSeidel.h"
#include "omp.h"
#include <iomanip>
#include <cstdlib>
#include <iostream>

using namespace std;

int Calc_par_block (double** u, double** f, int N, double eps) {

double max_err = 0; 
double h = 1.0 / (N + 1); 

int IterCnt = 0; 
int BlockSize = 20; 
int BlockCount; 

cout << fixed << setprecision(10);

if (N % BlockSize == 0) {
BlockCount = N / BlockSize;
int num_of_waves = BlockCount * 2 - 1;

do {
IterCnt++;
max_err = 0;

int num_of_elems_wave = 0;
int row_start = 0;
int col_start = -1;
for (int i_wave = 0; i_wave < num_of_waves; i_wave++) {
if (i_wave < BlockCount) {
num_of_elems_wave += 1;
col_start += 1;
}
else {
num_of_elems_wave -= 1;
row_start += 1;
}

if (col_start >= BlockCount) cout << "Error colstart" << endl;
if (row_start >= BlockCount) cout << "Error rowstart" << endl;

if (num_of_elems_wave == 0) cout << "Error num_of_elems_wave" << endl;

#pragma omp parallel shared(max_err)
#pragma omp for schedule(dynamic, 1)
for (int i_wave_elem = 0; i_wave_elem < num_of_elems_wave; i_wave_elem++) {
double max_err_wave = max_err;
for (int idx = 0; idx < BlockSize; idx++) 
for (int jdy = 0; jdy < BlockSize; jdy++) { 
int i = (row_start + i_wave_elem)*BlockSize + idx + 1;
int j = (col_start - i_wave_elem)*BlockSize + jdy + 1;

double u_old = u[i][j];

u[i][j] = 0.25*(u[i - 1][j] + u[i + 1][j] + u[i][j - 1] + u[i][j + 1] - h * h*f[i - 1][j - 1]);

double curr_err = abs(u[i][j] - u_old);

if (curr_err > max_err_wave) max_err_wave = curr_err;
}
#pragma omp critical
{
if (max_err_wave > max_err) max_err = max_err_wave;
}
}
}
} while (max_err > eps); 
}
else {
cout << "Error!!!" << endl;
}
return IterCnt;
}
