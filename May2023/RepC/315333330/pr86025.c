int i;
void
foo (int x)
{
if (x)
{
#pragma omp critical (foo)
i++;
}
else
{
#pragma omp critical
i++;
}
}
