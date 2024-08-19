void
foo (void)
{
int i, j;
#pragma omp for simd copyprivate(j	
for (i = 0; i < 16; ++i)		
;
}
