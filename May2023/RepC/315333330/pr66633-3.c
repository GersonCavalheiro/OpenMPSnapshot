void baz (int (*) ());
void
foo (void)
{
int i;
auto int bar (void) { return i; }
auto void bar2 (void)
{
#pragma omp parallel
baz (bar);
}
bar2 ();
}
