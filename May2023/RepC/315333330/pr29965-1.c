extern void baz () __attribute__ ((noreturn));
static inline void
foo ()
{
#pragma omp parallel
for (;;)
;
}
static inline void
bar ()
{
#pragma omp parallel
baz ();
}
void
foo1 ()
{
foo ();
}
void
foo2 ()
{
foo ();
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
