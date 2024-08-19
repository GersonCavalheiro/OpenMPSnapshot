struct S
{
#pragma omp declare simd linear(this)		
static void foo ();
void bar ();
};
void
S::bar ()
{
#pragma omp parallel firstprivate (this)	
;
#pragma omp parallel for lastprivate (this)	
for (int i = 0; i < 10; i++)
;
#pragma omp parallel shared (this)		
;
#pragma omp for linear (this)			
for (int i = 0; i < 10; i++)
;
#pragma omp task depend(inout: this)		
;
#pragma omp task depend(inout: this[0])	
;
#pragma omp parallel private (this)		
{
#pragma omp single copyprivate (this)	
;
}
}
template <int N>
struct T
{
#pragma omp declare simd linear(this)		
static void foo ();
void bar ();
};
template <int N>
void
T<N>::bar ()
{
#pragma omp parallel firstprivate (this)	
;
#pragma omp parallel for lastprivate (this)	
for (int i = 0; i < 10; i++)
;
#pragma omp parallel shared (this)		
;
#pragma omp for linear (this)			
for (int i = 0; i < 10; i++)
;
#pragma omp task depend(inout: this)		
;
#pragma omp task depend(inout: this[0])	
;
#pragma omp parallel private (this)		
{
#pragma omp single copyprivate (this)	
;
}
}
template struct T<0>;
