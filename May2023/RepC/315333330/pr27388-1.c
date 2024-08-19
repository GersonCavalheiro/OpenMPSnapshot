int n, o;
void
foo (void)
{
#pragma omp parallel firstprivate (n)
{
int i;
#pragma omp parallel for firstprivate (n)
for (i = 0; i < 10; i++)
++n;
#pragma omp atomic
o += n;
}
}
