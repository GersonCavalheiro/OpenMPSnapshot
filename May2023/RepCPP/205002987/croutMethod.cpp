#include <omp.h>
#include "croutMethod.h"
#include <cfloat>

#include "../../../lib/exceptions.h"

namespace numath {
namespace systemsOfEquations {

std::vector<double> croutMethod(std::vector<std::vector<double>> A, std::vector<double> b)
{
int N = A.size();
std::vector<double> results;
std::vector<std::vector<double>> L(N, std::vector<double>(N, 0.0));
std::vector<std::vector<double>> U(N, std::vector<double>(N, 0.0));
__initializeMatrix(L, U);

try
{
__LUFactoring(A, L, U, N);
std::vector<double> z = __forwardSubstitutionCM(L, b);
results = __backwardSubstitutionCM(U, z);
}
catch (DenominatorException &ex)
{
throw ex;
}
return results;
}

void __initializeMatrix(std::vector<std::vector<double>> &L, std::vector<std::vector<double>> &U)
{
int N = L.size();
#pragma omp parallel for
for (int i = 0; i < N; i++)
{
for (int j = 0; j < N; j++)
{
if (i < j)
{
U[i][j] = DBL_MAX;
L[i][j] = 0;
}
else if (i > j)
{
L[i][j] = DBL_MAX;
U[i][j] = 0;
}
else if (i == j)
{
U[i][j] = 1;
L[i][j] = DBL_MAX;
}
}
}
}

void __LUFactoring(std::vector<std::vector<double>> &A, std::vector<std::vector<double>> &L, std::vector<std::vector<double>> &U, int N)
{

for (int k = 1; k < N + 1; k++)
{
double sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int p = 0; p < k - 1; p++)
{
sum += L[k - 1][p] * U[p][k - 1];
}
L[k - 1][k - 1] = A[k - 1][k - 1] - sum;
for (int i = k + 1; i < N + 1; i++)
{
sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int p = 0; p < k - 1; p++)
{
sum += L[i - 1][p] * U[p][k - 1];
}
double denominator = U[k - 1][k - 1];
if (denominator == 0)
{
throw DenominatorException();
}
else
{
L[i - 1][k - 1] = (A[i - 1][k - 1] - sum) / U[k - 1][k - 1];
}
}

for (int j = k + 1; j < N + 1; j++)
{
sum = 0;

#pragma omp parallel for reduction(+:sum)
for (int p = 0; p < k - 1; p++)
{
sum += L[k - 1][p] * U[p][j - 1];
}
double denominator = L[k - 1][k - 1];
if (denominator == 0)
{
throw DenominatorException();
}
else
{
U[k - 1][j - 1] = (A[k - 1][j - 1] - sum) / L[k - 1][k - 1];
}
}
}
}

std::vector<double> __forwardSubstitutionCM(std::vector<std::vector<double>> &L, std::vector<double> &b)
{
int N = L.size();
std::vector<double> x(N, 0.0);
for (int i = 1; i < N + 1; i++)
{
double sum = 0;
for (int p = i - 1; p > 0; p--)
{
sum += (L[i - 1][p - 1] * x[p - 1]);
}
double denominator = L[i - 1][i - 1];
if (denominator == 0)
{
throw DenominatorException();
}
else
{
x[i - 1] = (b[i - 1] - sum) / L[i - 1][i - 1];
}
}
return x;
}

std::vector<double> __backwardSubstitutionCM(std::vector<std::vector<double>> &U, std::vector<double> &z)
{
int N = U.size();
std::vector<double> x(N, 0.0);
for (int i = N - 1; i >= 0; i--)
{
double sum = 0;
for (int j = i + 1; j < N; j++)
{
sum += U[i][j] * x[j];
}
double denominator = U[i][i];
if (denominator == 0)
{
throw DenominatorException();
}
else
{
x[i] = (z[i] - sum) / U[i][i];
}
}
return x;
}




}
}