void
foo ()
{
#pragma omp parallel
try
{
int q = 1;
}
catch (...)
{
}
}
