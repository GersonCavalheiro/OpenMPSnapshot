void
foo ()
{
#pragma omp parallel
try
{
}
catch (...)
{
int q = 1;
}
}
