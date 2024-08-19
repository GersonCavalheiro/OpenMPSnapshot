void
foo1 ()
{
#pragma omp sections
{
throw 0;
}
}
void
bar1 ()
{
#pragma omp sections
{
#pragma omp section
throw 0;
#pragma omp section
throw 0;
}
}
void
foo2 ()
{
#pragma omp sections
{
;
#pragma omp section
throw 0;
}
}
void
bar2 ()
{
#pragma omp sections
{
#pragma omp section
throw 0;
#pragma omp section
;
}
}
void
foo3 ()
{
#pragma omp parallel sections
{
throw 0;
}
}
void
bar3 ()
{
#pragma omp parallel sections
{
#pragma omp section
throw 0;
#pragma omp section
throw 0;
}
}
void
foo4 ()
{
#pragma omp parallel sections
{
;
#pragma omp section
throw 0;
}
}
void
bar4 ()
{
#pragma omp parallel sections
{
#pragma omp section
throw 0;
#pragma omp section
;
}
}
