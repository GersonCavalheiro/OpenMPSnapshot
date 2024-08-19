struct S { int i : 1; int j : 4; long long k : 25; };
void bar (struct S, int);
#pragma omp declare target to (bar)
void
foo (struct S a, struct S b, struct S c, struct S d)
{
#pragma omp target map (a)
bar (a, 0);
#pragma omp target map (a) map (b.i)		
bar (a, b.i);
#pragma omp target map (a) map (b.j)		
bar (a, b.j);
#pragma omp target map (a) map (b.k)		
bar (a, b.k);
#pragma omp target data map (a) map (b.i)	
{
#pragma omp target enter data map (alloc: a) map (to: c.j)		
#pragma omp target exit data map (release: a) map (from: d.k)	
}
}
