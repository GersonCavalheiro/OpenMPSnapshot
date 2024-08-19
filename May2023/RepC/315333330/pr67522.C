struct S;
template <int N>
void
foo (void)
{
#pragma omp simd linear (S)			
for (int i = 0; i < 16; i++)
;
#pragma omp target map (S[0:10])		
;
#pragma omp task depend (inout: S[0:10])	
;
#pragma omp for reduction (+:S[0:10])		
for (int i = 0; i < 16; i++)
;
}
void
bar ()
{
foo <0> ();
}
