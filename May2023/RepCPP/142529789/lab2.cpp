#include <iostream>
#include <fstream>
#include <math.h>
#include <iomanip>
#include <float.h>
#include <cstdlib>
#include <string.h>
#include <stdlib.h>
#include <sstream>
#include <omp.h>

using namespace std;
typedef double tdata;

const tdata PI = 3.141592653589793238462643;
const tdata PI2 = PI*PI;

const tdata K2 = 100.0;
const tdata L = 1.0;
const int n = 300;
const tdata h = L / n;

tdata cft1 = 1 / (h*h);
tdata cft2 = 4.0*cft1 + K2;
tdata eps = 1e-5;
const int ITER = 7000;

const int errtype = 2;

void Jacoby(int n, tdata **U, tdata **F);
void Seidel(int n, tdata **U, tdata **F);
void Black_and_White_Serial(int iter, int n, tdata **U, tdata **F);
void Black_and_White_Parallel(int iter, int n, tdata **U, tdata **F);

void zeroMtx(int n, tdata **A);
void printMatrix(int n, tdata **a);

tdata fRight(tdata x, tdata y);
tdata UPrecise(tdata x, tdata y);
void init(int n, tdata **A);
void CountNev(int n, tdata **U, tdata **f);

tdata MNormDlt(int n, tdata **A, tdata **B);
tdata MNorm(int n, tdata **A);

bool vyvod = true;

int main()
{
cout << "Hello, World!\n";

setlocale(LC_ALL, "Russian");

tdata *xGrid = new tdata[n + 1];
tdata *yGrid = new tdata[n + 1];

for (int i = 0; i <= n; i++)
{
xGrid[i] = i*h;
yGrid[i] = xGrid[i];
}

tdata **F = new tdata*[n + 1];
tdata **U = new tdata*[n + 1];
tdata **Up = new tdata*[n + 1];

for (int i = 0; i <= n; i++)
{
F[i] = new tdata[n + 1];
U[i] = new tdata[n + 1];
Up[i] = new tdata[n + 1];
}

for (int i = 0; i <= n; i++)
{
for (int j = 0; j <= n; j++)
{
F[i][j] = fRight(xGrid[i], yGrid[j]);
Up[i][j] = UPrecise(xGrid[i], yGrid[j]);
}
}

init(n + 1, U);

Black_and_White_Parallel(ITER, n, U, F);

cout << " error = " << MNormDlt(n + 1, U, Up) << endl;

ofstream solout("Solution.dat");

for (int i = 0; i <= n; i++)
for (int j = 0; j <= n; j++)
solout << xGrid[i] << " \t" << yGrid[j] << " \t" << U[i][j] << " \t" << Up[i][j] << " \t" << fabs(U[i][j] - Up[i][j]) << endl;

solout.close();

zeroMtx(n + 1, Up);
CountNev(n + 1, U, Up);
cout << "  " << MNormDlt(n + 1, F, Up) << endl;


cin.get();
cin.get();
return 0;
}

void Black_and_White_Serial(int iter, int n, tdata **U, tdata **F)
{
tdata **Y = new tdata*[n + 1];
for (int i = 0; i <= n; i++)
{
Y[i] = new tdata[n + 1];
for (int j = 0; j <= n; j++) Y[i][j] = U[i][j];
}

int k = 0;
int i = 0;
int j = 0;
int s = 1;

tdata error = 1.0;

{
tdata start_time = omp_get_wtime();
do
{
k++;

for (i = 1; i < n; i++)
{
(i % 2 == 0 ? s = 2 : s = 1);
for (j = s; j < n; j += 2)
Y[i][j] = (cft1*(Y[i - 1][j] + Y[i + 1][j] + Y[i][j - 1] + Y[i][j + 1]) + F[i][j]) / cft2;
}

for (i = 1; i < n; i++)
{
(i % 2 == 0 ? s = 1 : s = 2);
for (j = s; j < n; j += 2)
Y[i][j] = (cft1*(Y[i - 1][j] + Y[i + 1][j] + Y[i][j - 1] + Y[i][j + 1]) + F[i][j]) / cft2;
}




} 
while (k <= iter);

tdata end_time = omp_get_wtime();
cout << " -    " << k << " ." << endl;
cout << "  " << (end_time - start_time) << " ." << endl;
}

for (int i = 0; i <= n; i++)
{
for (int j = 0; j <= n; j++) U[i][j] = Y[i][j];
delete[] Y[i];
}
delete[] Y;
}

void Black_and_White_Parallel(int iter, int n, tdata **U, tdata **F)
{
tdata **Y = new tdata*[n + 1];
for (int i = 0; i <= n; i++)
{
Y[i] = new tdata[n + 1];
for (int j = 0; j <= n; j++) Y[i][j] = U[i][j];
}

int k = 0;
int i = 0;
int j = 0;
int s = 1;

#pragma omp parallel shared(n)
{
tdata start_time = omp_get_wtime();
do
{
#pragma omp for private(i, j, s) 
for (i = 1; i < n; i++)
{
(i % 2 == 0 ? s = 2 : s = 1);
for (j = s; j < n; j += 2)
Y[i][j] = (cft1*(Y[i - 1][j] + Y[i + 1][j] + Y[i][j - 1] + Y[i][j + 1]) + F[i][j]) / cft2;
}

#pragma omp for private(i, j, s) 
for (i = 1; i < n; i++)
{
(i % 2 == 0 ? s = 1 : s = 2);
for (j = s; j < n; j += 2)
Y[i][j] = (cft1*(Y[i - 1][j] + Y[i + 1][j] + Y[i][j - 1] + Y[i][j + 1]) + F[i][j]) / cft2;
}

k++;
} while (k <= iter);

tdata end_time = omp_get_wtime();
cout << "  " << (end_time - start_time) << " ." << endl;
}

for (int i = 0; i <= n; i++)
{
for (int j = 0; j <= n; j++) U[i][j] = Y[i][j];
delete[] Y[i];
}
delete[] Y;
}

void Jacoby(int n, tdata **U, tdata **F)
{
tdata error = 1.0;

tdata **Y = new tdata*[n + 1];
tdata **f = new tdata*[n + 1];
for (int i = 0; i <= n; i++)
{
Y[i] = new tdata[n + 1];
f[i] = new tdata[n + 1];
}

init(n + 1, Y);
zeroMtx(n + 1, f);
cout.precision(16);

int k = 0;
double r = MNorm(n + 1, F);

tdata start_time = clock();

do
{
k++;
for (int i = 1; i < n; i++)
for (int j = 1; j < n; j++)
Y[i][j] = U[i][j];

for (int i = 1; i < n; i++)
for (int j = 1; j < n; j++)
U[i][j] = (cft1*(Y[i - 1][j] + Y[i + 1][j] + Y[i][j - 1] + Y[i][j + 1]) + F[i][j]) / cft2;




} while (k <= ITER);

tdata end_time = clock();

cout << "    " << k << " ." << endl;
cout << "  " << (end_time - start_time) / CLOCKS_PER_SEC << " ." << endl;

for (int i = 0; i <= n; i++)
{
delete[] Y[i];
delete[] f[i];
}
delete[] Y;
delete[] f;
}

void Seidel(int n, tdata **U, tdata **F)
{
tdata error = 1.0;

tdata **Y = new tdata*[n + 1];
tdata **f = new tdata*[n + 1];
for (int i = 0; i <= n; i++)
{
Y[i] = new tdata[n + 1];
f[i] = new tdata[n + 1];
}

init(n + 1, Y);
zeroMtx(n + 1, f);
cout.precision(16);

int k = 0;

tdata start_time = clock();

do
{
k++;
for (int i = 1; i < n; i++)
for (int j = 1; j < n; j++)
Y[i][j] = U[i][j];

for (int i = 1; i < n; i++)
for (int j = 1; j < n; j++)
U[i][j] = (cft1*(U[i - 1][j] + Y[i + 1][j] + U[i][j - 1] + Y[i][j + 1]) + F[i][j]) / cft2;




error = MNormDlt(n + 1, Y, U);
if (vyvod) cout << "err = " << error << endl;
} while (k <= ITER);

tdata end_time = clock();

cout << "    " << k << " ." << endl;
cout << "  " << (end_time - start_time) / CLOCKS_PER_SEC << " ." << endl;

for (int i = 0; i <= n; i++)
{
delete[] Y[i];
delete[] f[i];
}
delete[] Y;
delete[] f;
}

void zeroMtx(int n, tdata **A)
{
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
A[i][j] = 0.0;
}
}

void init(int n, tdata **A)
{
zeroMtx(n, A);
for (int i = 1; i < n - 1; i++)
{
for (int j = 1; j < n - 1; j++)
A[i][j] = 0.5;
}
}

void CountNev(int n, tdata **U, tdata **f)
{
zeroMtx(n, f);

for (int i = 1; i < n - 1; i++)
for (int j = 1; j < n - 1; j++)
f[i][j] = -cft1*(U[i - 1][j] + U[i + 1][j] + U[i][j + 1] + U[i][j - 1] - 4.0*U[i][j]) + K2*U[i][j];
}

tdata UPrecise(tdata x, tdata y)
{
return (x - x*x)*sin(PI*y);
}

void printMatrix(int n, tdata **a)
{
for (int i = 0; i < n; i++)
{
for (int j = 0; j < n; j++)
cout << a[i][j] << " \t";
cout << endl;
}
cout << endl;
}

tdata fRight(tdata x, tdata y)
{
return sin(PI*y)*((x - x*x)*(K2 + PI*PI) + 2.0);
}

tdata MNormDlt(int n, tdata **A, tdata **B)
{
tdata err = 0.0;

if (errtype == 2)
{
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
err += pow((A[i][j] - B[i][j]), 2);


err /= cft1;

err = sqrt(err);
}
else
{
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
if (fabs(A[i][j] - B[i][j]) > err) err = fabs(A[i][j] - B[i][j]);
}

return err;
}

tdata MNorm(int n, tdata **A)
{
tdata err = 0.0;

if (errtype == 2)
{
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
err += pow(A[i][j], 2);


err /= cft1;
err = sqrt(err);
}
else
{
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
if (fabs(A[i][j]) > err) err = fabs(A[i][j]);
}

return err;
}