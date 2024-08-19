void f1(void)
{
#pragma omp flush
}
int x;
void f2(bool p)
{
int z;
if (p)
{
#pragma omp flush (x)
}
else
{
#pragma omp flush (x, z, p)
}
}
