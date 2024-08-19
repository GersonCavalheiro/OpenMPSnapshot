void
foo (int i)
{
extern int a[i];	
static int b[i];	
#pragma omp parallel
{
a[0] = 0;
b[0] = 0;
}
#pragma omp parallel shared (a, b)
{
a[0] = 0;
b[0] = 0;
}
#pragma omp parallel private (a, b)
{
a[0] = 0;
b[0] = 0;
}
#pragma omp parallel firstprivate (a, b)
{
a[0] = 0;
b[0] = 0;
}
}
