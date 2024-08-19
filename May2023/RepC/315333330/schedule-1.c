void
foo (void)
{
int i;
#pragma omp for schedule(static, 1)
for (i = 0; i < 10; i++)
;
#pragma omp for schedule(static, 0)		
for (i = 0; i < 10; i++)
;
#pragma omp for schedule(static, -7)		
for (i = 0; i < 10; i++)
;
}
