void f1(void)
{
#pragma omp barrier
}
void f2(bool p)
{
if (p)
{
#pragma omp barrier
}
}
