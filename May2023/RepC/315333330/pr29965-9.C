void
foo1 ()
{
#pragma omp single
throw 0;
}
void
foo2 ()
{
#pragma omp master
throw 0;
}
void
foo3 ()
{
#pragma omp ordered
throw 0;
}
void
foo4 ()
{
#pragma omp critical
throw 0;
}
