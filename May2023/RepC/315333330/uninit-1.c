void foo()
{
int i;
#pragma omp parallel shared(i)
{
i = 0;
++i;
}
}
