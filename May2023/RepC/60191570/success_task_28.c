void foo_multidep(int *p, int M, int N)
{
#pragma omp task inout({((int (*)[N])p)[i][j], i=0:M-1, j=0:N-1})
{}
}
void foo_normaldep(int *p, int M, int N)
{
#pragma omp task inout(((int (*)[N])p)[0:M])
{}
}
