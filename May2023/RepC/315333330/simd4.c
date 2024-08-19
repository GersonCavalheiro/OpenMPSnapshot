struct S *p;	
float f;
int j;
void
foo (void)
{
#pragma omp simd linear(p) linear(f : 1)
for (int i = 0; i < 10; i++)
;
#pragma omp simd linear(j : 7.0)	
for (int i = 0; i < 10; i++)
;
}
