#define N 1024
int
foo (int *__restrict__ ap)
{
int *bp = ap;
#pragma omp parallel for
for (unsigned int idx = 0; idx < N; idx++)
ap[idx] = bp[idx];
}
