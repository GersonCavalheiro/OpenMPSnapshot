extern void baz (void) __attribute__((noreturn));
void
foo (void)
{
#pragma omp parallel
for (;;)
;
}
void
bar (void)
{
#pragma omp parallel
baz ();
}
