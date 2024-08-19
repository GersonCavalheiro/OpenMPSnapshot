#define p		_Pragma ("omp parallel")
#define omp_p		_Pragma ("omp p")
void bar (void);
void
foo (void)
{
#pragma omp p		
bar ();
omp_p			
bar ();
}
#define parallel	serial
#define omp_parallel	_Pragma ("omp parallel")
void
baz (void)
{
#pragma omp parallel	
bar ();
omp_parallel		
bar ();
}
