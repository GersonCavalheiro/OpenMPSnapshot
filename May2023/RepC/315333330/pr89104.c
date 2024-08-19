#pragma omp declare simd uniform (x) aligned (x)
int
foo (int *x, int y)
{
return x[y];
}
