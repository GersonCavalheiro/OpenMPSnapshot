int main()
{
int i;
#pragma omp parallel for firstprivate(i)
for (i = 0; i  < 100; ++i)
{
}
}
