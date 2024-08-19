struct A { A () : i (0) {} int i; };
void
foo ()
{
A a;
#pragma omp declare reduction (+: A: omp_out.i += omp_in.i)
#pragma omp parallel reduction (+: a)
;
}
