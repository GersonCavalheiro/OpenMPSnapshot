struct A
{
~A () throw ();
};
void foo (A);
void
bar ()
{
#pragma omp parallel
foo (A ());
}
