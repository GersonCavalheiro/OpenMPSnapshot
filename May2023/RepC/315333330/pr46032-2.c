#define N 2
int
foo (void)
{
int a[N], b[N], c[N];
int *ap = &a[0];
int *bp = &b[0];
int *cp = &c[0];
#pragma omp parallel for
for (unsigned int idx = 0; idx < N; idx++)
{
ap[idx] = 1;
bp[idx] = 2;
cp[idx] = ap[idx];
}
return *cp;
}
