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

double **mult_LU(int n, double **lu);
double error(int n, double **a, double **pr);
double **LU_omp(double **a, int n);
double **LU_ser(double **a, int n);
void nullMatrix(double **mass, int n, int m);
void nullRow(double *mass, int n);
void printMatrix(double **a, int n, int m);
void printColumn(double *a, int n);
void fullGauss(int n, double **M, double *r);
double **Blocks(int size, int r, double **A);
double **Blocks1(int size, int r, double **A);
double **Blocks_omp(int size, int r, double **A);
void MtxDivide(int size, int r, double **a11, double **a12, double **a21, double **a22, double **a);
void MtxSplit(int size, int r, double **m11, double **m12, double **m21, double **m22, double **mtx);
double **mtxPlus(int n, double **A, double **B);
double **mtxMinus(int n, double **A, double **B);


int main()
{
cout.precision(16);

int n;
cout << "n = ";
cin >> n;
cin.get();

double **a = new double*[n];
for (int j = 0; j < n; j++)
a[j] = new double[n];
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
a[i][j] = rand() % 20 - 10;

double **lup = new double*[n];
double **lus = new double*[n];
double **lubls = new double*[n];
double **lublp = new double*[n];
for (int j = 0; j < n; j++)
{
lup[j] = new double[n];
lus[j] = new double[n];
lubls[j] = new double[n];
lublp[j] = new double[n];
}

double start_time = clock();
lus = LU_ser(a, n);
double end_time = clock();
cout << "Time serial = " << (end_time - start_time) / CLOCKS_PER_SEC << endl;


double **pr = new double*[n];
for (int j = 0; j < n; j++)
pr[j] = new double[n];
pr = mult_LU(n, lus);

cout << "Error serial = " << error(n, a, pr) << endl;

start_time = clock();
lup = LU_omp(a, n);
end_time = clock();
cout << "Time parallel = " << (end_time - start_time) / CLOCKS_PER_SEC << endl;

nullMatrix(pr, n, n);
pr = mult_LU(n, lup);

cout << "Error parallel = " << error(n, a, pr) << endl;

int r;
cout << "r = ";
cin >> r;
cin.get();




start_time = clock();
lublp = Blocks1(n, r, a);
end_time = clock();
cout << "Time block parallel= " << (end_time - start_time) / CLOCKS_PER_SEC << endl;

nullMatrix(pr, n, n);
pr = mult_LU(n, lublp);

cout << "Error block parallel = " << error(n, a, pr) << endl;

cin.get();

for (int i = 0; i < n; i++)
{
delete[] a[i];
delete[] lus[i];
delete[] lup[i];
delete[] pr[i];
}

delete[] a;
delete[] lus;
delete[] lup;
delete[] pr;

return 0;
}

double **mult_LU(int n, double **lu)
{
double **pr = new double*[n];
for (int i = 0; i < n; i++)
pr[i] = new double[n];
nullMatrix(pr, n, n);

pr[0][0] = lu[0][0];

for (int m = 1; m < n; m++)
{
pr[0][m] = lu[0][m];
pr[m][0] = lu[m][0] * lu[0][0];
pr[m][m] = lu[m][m];
for (int k = 0; k < m; k++)
pr[m][m] += lu[m][k] * lu[k][m];
}

for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
{
if ((i < j) && (i != 0))
{
pr[i][j] = lu[i][j];
for (int k = 0; k < i; k++)
pr[i][j] += lu[i][k] * lu[k][j];
}
else if ((i > j) && (j != 0))
for (int k = 0; k <= j; k++)
pr[i][j] += lu[i][k] * lu[k][j];
}

return pr;
}


double error(int n, double **a, double **pr)
{
double **err = new double*[n];
for (int i = 0; i < n; i++)
err[i] = new double[n];
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
err[i][j] = fabs(a[i][j] - pr[i][j]);

double errmax, err1;
errmax = err[0][0];
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
{
err1 = err[i][j];
if (errmax <= err1)
errmax = err1;
}

return errmax;
}

double **LU_omp(double **a, int n)
{
double **C = new double*[n];
for (int i = 0; i < n; i++)
{
C[i] = new double[n];
for (int j = 0; j < n; j++)
C[i][j] = a[i][j];
}

int i, j, k;

for (k = 0; k < n - 1; ++k)
{
#pragma omp parallel for shared(a,n,k) private(i,j)
for (i = k + 1; i < n; i++)
{
C[i][k] /= C[k][k];
for (j = k + 1; j < n; j++)
{
C[i][j] -= C[i][k] * C[k][j];
}
}
}

return C;
}

double **LU_ser(double **a, int n)
{
int i, j, k;

double **C = new double *[n];
for (i = 0; i < n; i++)
{
C[i] = new double[n];
for (j = 0; j < n; j++)
C[i][j] = a[i][j];
}

for (k = 0; k < n - 1; ++k)
{
for (i = k + 1; i < n; i++)
{
C[i][k] /= C[k][k];
for (j = k + 1; j < n; j++)
{
C[i][j] -= C[i][k] * C[k][j];
}
}
}

return C;
}

void nullMatrix(double **mass, int n, int m)
{
for (int i = 0; i < n; i++)
for (int j = 0; j < m; j++)
mass[i][j] = 0.;

return;
}

void nullRow(double *mass, int n)
{
for (int i = 0; i < n; i++)
mass[i] = 0.;

return;
}

void printMatrix(double **a, int n, int m)
{
for (int i = 0; i < n; i++)
{
for (int j = 0; j < m; j++)
cout << a[i][j] << " ";
cout << endl;
}
cout << endl;
return;
}

void printColumn(double *a, int n)
{
for (int i = 0; i < n; i++)
cout << a[i] << endl;
return;
}

void fullGauss(int n, double **M, double *r)
{

double *temp = new double[n];
double t = 0.0;
double **A = new double *[n];
double *f = new double[n];

for (int i = 0; i < n; i++)
{
A[i] = new double[n];
for (int j = 0; j < n; j++)
{
A[i][j] = M[i][j];
}
f[i] = r[i];
}

for (int i = 0; i < n - 1; i++)
{
t = A[i][i];
for (int j = i; j < n; j++)
A[i][j] /= t;
f[i] /= t;

for (int j = i + 1; j < n; j++)
{
t = A[j][i];
for (int k = i; k < n; k++)
A[j][k] -= t * A[i][k];
f[j] -= t * f[i];
}
}
t = A[n - 1][n - 1];
f[n - 1] /= t;
A[n - 1][n - 1] /= t;

for (int j = n - 1; j > 0; j--)
{
for (int i = j - 1; i >= 0; i--)
{
f[i] -= f[j] * A[i][j];
}
}

for (int i = 0; i < n; i++)
r[i] = f[i];

for (int i = 0; i < n; i++)
{
delete[] A[i];
}
delete[] A;
delete[] temp;
delete[] f;
}

double **Blocks(int size, int r, double **A)
{
double **C = new double*[size];
for (int i = 0; i < size; i++)
{
C[i] = new double[size];
}

if (size <= r)
{
C = LU_ser(A, size);
}
else
{
int n = size;

double **a11 = new double *[r];
double **lu11 = new double *[r];
double **l11 = new double *[r];
double **u11t = new double *[r];

double *x = new double[r];

for (int i = 0; i < r; i++)
{
a11[i] = new double[r];
lu11[i] = new double[r];
u11t[i] = new double[r];
l11[i] = new double[r];
}

double **a12 = new double *[r];
double **lu12 = new double *[r];
for (int i = 0; i < r; i++)
{
a12[i] = new double[n - r];
lu12[i] = new double[n - r];
}

double **a21 = new double *[n - r];
double **lu21 = new double *[n - r];
for (int i = 0; i < n - r; i++)
{
a21[i] = new double[r];
lu21[i] = new double[r];
}

double **a22 = new double *[n - r];
double **aa22 = new double *[n - r];
double **lu22 = new double *[n - r];
for (int i = 0; i < n - r; i++)
{
a22[i] = new double[n - r];
aa22[i] = new double[n - r];
lu22[i] = new double[n - r];
}

MtxDivide(n, r, a11, a12, a21, a22, A);

lu11 = LU_ser(a11, r);

nullMatrix(u11t, r, r);
nullMatrix(l11, r, r);
for (int i = 0; i < r; i++)
{
l11[i][i] = 1.;
for (int j = 0; j < i; j++)
l11[i][j] = lu11[i][j];
for (int j = i; j < r; j++)
u11t[j][i] = lu11[i][j];
}

for (int j = 0; j < n - r; j++)
{
for (int i = 0; i < r; i++)
x[i] = a12[i][j];
fullGauss(r, l11, x);
for (int i = 0; i < r; i++)
lu12[i][j] = x[i];
}

nullRow(x, r);

for (int j = 0; j < n - r; j++)
{
for (int i = 0; i < r; i++)
x[i] = a21[j][i];
fullGauss(r, u11t, x);
for (int s = 0; s < r; s++)
lu21[j][s] = x[s];
}

for (int i = 0; i < n - r; i++)
for (int j = 0; j < n - r; j++)
{
double sum = 0.;
for (int s = 0; s < r; s++)
sum += lu21[i][s] * lu12[s][j];
aa22[i][j] = a22[i][j] - sum;
}

n -= r;
nullMatrix(lu22, n, n);
lu22 = Blocks(n, r, aa22);
MtxSplit(n + r, r, lu11, lu12, lu21, lu22, C);

for (int i = 0; i < r; i++)
{
delete[] l11[i];        delete[] lu11[i];         delete[] u11t[i];
delete[] a12[i];        delete[] lu12[i];
}
delete[] l11;               delete[] lu11;            delete[] u11t;
delete[] a12;               delete[] lu12;            delete[] x;

for (int i = 0; i < n - r; i++)
{
delete[] a22[i];        delete[] aa22[i];         delete[] lu22[i];
delete[] lu21[i];       delete[] a21[i];
}
delete[] a22;               delete[] a21;             delete[] lu22;
delete[] aa22;              delete[] lu21;
}

return C;
}

double **Blocks1(int size, int r, double **A)
{
int count = 0;

double **C = new double*[size];
for (int i = 0; i < size; i++)
{
C[i] = new double[size];
}

if (size <= r)
{
C = LU_ser(A, size);
}
else
{
int n = size;



C = LU_ser(A, r);



double **l11 = new double *[r];
double **u11t = new double *[r];
double *x = new double[r];

for (int i = 0; i < r; i++)
{
u11t[i] = new double[r];
l11[i] = new double[r];
}

for (int i = 0; i < r; i++)
{
l11[i][i] = 1.;
for (int j = 0; j < i; j++)
l11[i][j] = C[i][j];
for (int j = i; j < r; j++)
u11t[j][i] = C[i][j];
}

for (int j = 0; j < n - r; j++)
{
for (int i = 0; i < r; i++)
x[i] = A[i][j + r];
fullGauss(r, l11, x);
for (int i = 0; i < r; i++)
C[i][j + r] = x[i];
}

nullRow(x, r);

for (int j = 0; j < n - r; j++)
{
for (int i = 0; i < r; i++)
x[i] = A[j + r][i];
fullGauss(r, u11t, x);
C[j + r] = x;
}

double **aa22 = new double*[n - r];
for (int i = 0; i < n - r; i++)
aa22[i] = new double[n - r];

for (int i = 0; i < n - r; i++)
for (int j = 0; j < n - r; j++)
{
double sum = 0.;
for (int s = 0; s < r; s++)
sum += C[i + r][s] * C[s][j + r];
aa22[i][j] = C[i + r][j + r] - sum;
}

n -= r;

double **lu22 = new double*[n];
for (int i = 0; i < n; i++)
lu22[i] = new double[n];

nullMatrix(lu22, n, n);
lu22 = Blocks1(n, r, aa22);
for (int i = 0; i < n; i++)
for (int j = 0; j < n; j++)
C[i + r][j + r] = lu22[i][j];

for (int i = 0; i < r; i++)
{
delete[] l11[i];          delete[] u11t[i];
}
delete[] l11;                 delete[] u11t;                delete[] x;

for (int i = 0; i < n - r; i++)
{
delete[] aa22[i];         delete[] lu22[i];
}
delete[] lu22;                delete[] aa22;
}

return C;
}

double **Blocks_omp(int size, int r, double **A)
{
double **C = new double*[size];
for (int i = 0; i < size; i++)
{
C[i] = new double[size];
}

if (size <= r)
{
C = LU_omp(A, size);
}
else
{
int n = size;

double **a11 = new double *[r];
double **lu11 = new double *[r];
double **l11 = new double *[r];
double **u11t = new double *[r];

double *x = new double[r];

for (int i = 0; i < r; i++)
{
a11[i] = new double[r];
lu11[i] = new double[r];
u11t[i] = new double[r];
l11[i] = new double[r];
}

double **a12 = new double *[r];
double **lu12 = new double *[r];
for (int i = 0; i < r; i++)
{
a12[i] = new double[n - r];
lu12[i] = new double[n - r];
}

double **a21 = new double *[n - r];
double **lu21 = new double *[n - r];
for (int i = 0; i < n - r; i++)
{
a21[i] = new double[r];
lu21[i] = new double[r];
}

double **a22 = new double *[n - r];
double **aa22 = new double *[n - r];
double **lu22 = new double *[n - r];
for (int i = 0; i < n - r; i++)
{
a22[i] = new double[n - r];
aa22[i] = new double[n - r];
lu22[i] = new double[n - r];
}

MtxDivide(n, r, a11, a12, a21, a22, A);

lu11 = LU_omp(a11, r);

nullMatrix(u11t, r, r);
nullMatrix(l11, r, r);
for (int i = 0; i < r; i++)
{
l11[i][i] = 1.;
for (int j = 0; j < i; j++)
l11[i][j] = lu11[i][j];
for (int j = i; j < r; j++)
u11t[j][i] = lu11[i][j];
}

int l, ll = 0;
#pragma omp parallel for shared(a12,n,r,lu12) private(x,l,ll)
for (l = 0; l < n - r; l++)
{
for (ll = 0; ll < r; ll++)
x[ll] = a12[ll][l];
fullGauss(r, l11, x);
for (ll = 0; ll < r; ll++)
lu12[ll][l] = x[ll];
}

nullRow(x, r);

#pragma omp parallel for shared(a21,n,r,lu21) private(x,l,ll)
for (l = 0; l < n - r; l++)
{
for (ll = 0; ll < r; ll++)
x[ll] = a21[l][ll];
fullGauss(r, u11t, x);
for (ll = 0; ll < r; ll++)
lu21[l][ll] = x[ll];
}

for (int i = 0; i < n - r; i++)
for (int j = 0; j < n - r; j++)
{
double sum = 0.;
for (int s = 0; s < r; s++)
sum += lu21[i][s] * lu12[s][j];
aa22[i][j] = a22[i][j] - sum;
}

n -= r;
nullMatrix(lu22, n, n);
lu22 = Blocks_omp(n, r, aa22);
MtxSplit(n + r, r, lu11, lu12, lu21, lu22, C);

for (int i = 0; i < r; i++)
{
delete[] l11[i];        delete[] lu11[i];         delete[] u11t[i];
delete[] a12[i];        delete[] lu12[i];
}
delete[] l11;               delete[] lu11;            delete[] u11t;
delete[] a12;               delete[] lu12;            delete[] x;

for (int i = 0; i < n - r; i++)
{
delete[] a22[i];        delete[] aa22[i];         delete[] lu22[i];
delete[] lu21[i];       delete[] a21[i];
}
delete[] a22;               delete[] a21;             delete[] lu22;
delete[] aa22;              delete[] lu21;
}

return C;
}


void MtxDivide(int size, int r, double **a11, double **a12, double **a21, double **a22, double **a)
{
for (int i = 0; i < r; i++)
for (int j = 0; j < r; j++)
a11[i][j] = a[i][j];

for (int i = 0; i < r; i++)
for (int j = 0; j < size - r; j++)
a12[i][j] = a[i][j + r];

for (int i = 0; i < r; i++)
for (int j = 0; j < size - r; j++)
a21[j][i] = a[j + r][i];

for (int i = 0; i < size - r; i++)
for (int j = 0; j < size - r; j++)
a22[i][j] = a[i + r][j + r];
}

void MtxSplit(int size, int r, double **m11, double **m12, double **m21, double **m22, double **mtx)
{
for (int i = 0; i < r; i++)
for (int j = 0; j < r; j++)
{
mtx[i][j] = m11[i][j];
}

for (int i = 0; i < size - r; i++)
for (int j = 0; j < size - r; j++)
{
mtx[i + r][j + r] = m22[i][j];
}

for (int i = 0; i < r; i++)
for (int j = 0; j < size - r; j++)
{
mtx[i][j + r] = m12[i][j];
mtx[j + r][i] = m21[j][i];
}
}

double **mtxPlus(int n, double **A, double **B)
{
double **result = new double*[n];
for (int i = 0; i < n; i++)
{
result[i] = new double[n];
for (int j = 0; j < n; j++)
{
result[i][j] = A[i][j] + B[i][j];
}
}
return result;
}

double **mtxMinus(int n, double **A, double **B)
{
double **result = new double*[n];
for (int i = 0; i < n; i++)
{
result[i] = new double[n];
for (int j = 0; j < n; j++)
{
result[i][j] = A[i][j] - B[i][j];
}
}
return result;
}