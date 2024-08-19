void
foo (void)
{
int i;
#pragma omp taskloop
for (i = 0; i < 64; i++)
{
int j;
#pragma omp for			
for (j = 0; j < 10; j++)
;
#pragma omp single		
;
#pragma omp sections		
{
#pragma omp section
;
}
#pragma omp barrier		
#pragma omp master		
;
#pragma omp ordered		
;
#pragma omp ordered threads	
;
#pragma omp ordered simd threads	
;
#pragma omp simd
for (j = 0; j < 10; j++)
#pragma omp ordered simd
;
#pragma omp critical
{
#pragma omp simd
for (j = 0; j < 10; j++)
#pragma omp ordered simd
;
}
}
#pragma omp taskloop
for (i = 0; i < 64; i++)
#pragma omp parallel
{
int j;
#pragma omp for
for (j = 0; j < 10; j++)
;
#pragma omp single
;
#pragma omp sections
{
#pragma omp section
;
}
#pragma omp barrier
#pragma omp master
;
#pragma omp ordered		
;
#pragma omp ordered threads	
;
#pragma omp simd
for (j = 0; j < 10; j++)
#pragma omp ordered simd
;
#pragma omp critical
{
#pragma omp simd
for (j = 0; j < 10; j++)
#pragma omp ordered simd
;
}
}
#pragma omp taskloop
for (i = 0; i < 64; i++)
#pragma omp target
{
int j;
#pragma omp for
for (j = 0; j < 10; j++)
;
#pragma omp single
;
#pragma omp sections
{
#pragma omp section
;
}
#pragma omp barrier
#pragma omp master
;
#pragma omp ordered		
;
#pragma omp ordered threads	
;
#pragma omp simd
for (j = 0; j < 10; j++)
#pragma omp ordered simd
;
#pragma omp critical
{
#pragma omp simd
for (j = 0; j < 10; j++)
#pragma omp ordered simd
;
}
}
#pragma omp ordered
{
#pragma omp ordered			
;
}
#pragma omp ordered threads
{
#pragma omp ordered			
;
}
#pragma omp ordered
{
#pragma omp ordered threads		
;
}
#pragma omp ordered threads
{
#pragma omp ordered threads		
;
}
#pragma omp critical
{
#pragma omp ordered simd		
;
}
#pragma omp for ordered
for (i = 0; i < 64; i++)
#pragma omp parallel
{
#pragma omp ordered threads	
;
}
#pragma omp for ordered
for (i = 0; i < 64; i++)
#pragma omp parallel
{
#pragma omp ordered		
;
}
#pragma omp for ordered(1)
for (i = 0; i < 64; i++)
#pragma omp parallel
{
#pragma omp ordered depend(source)	
#pragma omp ordered depend(sink: i - 1)	
}
}
