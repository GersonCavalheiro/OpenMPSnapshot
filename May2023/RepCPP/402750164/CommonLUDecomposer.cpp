#include "../header.hpp"

CommonLUDecomposer::CommonLUDecomposer(int sizeParam)
{
size = sizeParam;
a = new double[size * size];
}

CommonLUDecomposer::~CommonLUDecomposer()
{
delete[] a;
}

void CommonLUDecomposer::setMatrix(double *x)
{
for (int i = 0; i < size * size; i++)
a[i] = x[i];
}


void CommonLUDecomposer::findDecomposition()
{
for (int i = 0; i < size; i++)
for (int j = i + 1; j < size; j++)
{
a[j * size + i] /= a[i * size + i];
for (int k = i + 1; k < size; k++)
a[j * size + k] -= a[j * size + i] * a[i * size + k];
}
}


void CommonLUDecomposer::findDecompositionParallel(int numTh)
{
int i, j, k;
for (i = 0; i < size; i++)
{
for (j = i + 1; j < size; j++)
{
a[j * size + i] /= a[i * size + i];
#pragma omp parallel for private(k) num_threads(numTh)
for (k = i + 1; k < size; k++)
{
a[j * size + k] -= a[j * size + i] * a[i * size + k];
}
}
}
}

void CommonLUDecomposer::printDecomposition()
{
cout << "Demmel Lu decomposition" << endl;
for (int i = 0; i < size; i++)
{
for (int j = 0; j < size; j++)
{
cout << a[i * size + j] << " ";
}
cout << endl;
}
}