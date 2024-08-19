#define omp		FOO
#define p		parallel
#define omp_parallel	_Pragma ("omp parallel")
#define omp_p		_Pragma ("omp p")
void bar (void);
void
foo (void)
{
#pragma omp parallel
bar ();
#pragma omp p
bar ();
omp_parallel
bar ();
omp_p
bar ();
}
