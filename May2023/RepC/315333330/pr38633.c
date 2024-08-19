void
foo ()
{
#pragma omp parallel
{
struct A { int i; } j;
j.i = 6;
j.i++;
}
}
