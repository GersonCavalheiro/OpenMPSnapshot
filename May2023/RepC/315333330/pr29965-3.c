extern void baz () __attribute__ ((noreturn));
void
foo1 ()
{
#pragma omp single
for (;;);
}
void
bar1 ()
{
#pragma omp single
baz ();
}
void
foo2 ()
{
#pragma omp master
for (;;);
}
void
bar2 ()
{
#pragma omp master
baz ();
}
void
foo3 ()
{
#pragma omp ordered
for (;;);
}
void
bar3 ()
{
#pragma omp ordered
baz ();
}
void
foo4 ()
{
#pragma omp critical
for (;;);
}
void
bar4 ()
{
#pragma omp critical
baz ();
}
