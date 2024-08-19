struct S { int s; } s;
void
foo (void)
{
#pragma omp parallel
{
#pragma omp cancel parallel if (s)	
}
}
