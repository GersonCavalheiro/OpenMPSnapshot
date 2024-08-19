template <class T> void
foo (T x)
{
#pragma omp parallel
{
#pragma omp cancel parallel if (x)	
}
}
struct S {};
void
bar ()
{
S s;
foo (s);
}
