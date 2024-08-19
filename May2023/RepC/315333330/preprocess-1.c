void foo (void)
{
int i1, j1, k1;
#define p parallel
#define P(x) private (x##1)
#define S(x) shared (x##1)
#define F(x) firstprivate (x##1)
#pragma omp p P(i) S(j) F(k)
;
}
