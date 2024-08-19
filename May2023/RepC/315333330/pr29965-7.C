void
foo ()
{
#pragma omp parallel
throw 0;
}
static inline void
bar ()
{
#pragma omp parallel
throw 0;
}
void
bar1 ()
{
bar ();
}
void
bar2 ()
{
bar ();
}
