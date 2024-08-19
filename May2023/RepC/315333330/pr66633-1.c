void baz (int (*) ());
void
foo (void)
{
int i;
auto int bar (void) { return i; }
#pragma omp parallel
baz (bar);
}
