void
f1 (int *x)
{
int i;
#pragma omp simd
for (i = 0; i < 100; i++)
#pragma omp ordered simd
x[i / 2] = i;
}
void
f2 (int *x)
{
int i;
#pragma omp parallel for simd ordered
for (i = 0; i < 100; i++)
#pragma omp ordered threads simd
x[i / 2] = i;
}
void
f3 (int *x)
{
int i;
#pragma omp parallel for ordered
for (i = 0; i < 100; i++)
#pragma omp ordered
x[i / 2] = i;
}
void
f4 (int *x)
{
int i;
#pragma omp parallel for ordered
for (i = 0; i < 100; i++)
#pragma omp ordered threads
x[i / 2] = i;
}
void
f5 (int n, int ***x)
{
int i, j, k;
#pragma omp parallel for ordered(3)
for (i=0; i < n; i++)
for (j=0; j < n; ++j)
for (k=0; k < n; ++k)
{
#pragma omp ordered depend(sink:i-8,j-2,k+2) depend(sink:i, j-1,k) depend(sink:i-4,j-3,k+6) depend(sink:i-6,j-4,k-6)
x[i][j][k] = i + j + k;
#pragma omp ordered depend(source)
}
}
