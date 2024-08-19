void
f1 (void)
{
_Atomic int i;
#pragma omp for		
for (i = 0; i < 64; i++)
;
#pragma omp parallel for	
for (i = 0; i < 64; i++)
;
#pragma omp simd		
for (i = 0; i < 64; i++)
;
#pragma omp parallel for simd	
for (i = 0; i < 64; i++)
;
#pragma omp for simd		
for (i = 0; i < 64; i++)
;
#pragma omp for		
for (_Atomic int j = 0; j < 64; j++)
;
#pragma omp parallel for	
for (_Atomic int j = 0; j < 64; j++)
;
#pragma omp simd		
for (_Atomic int j = 0; j < 64; j++)
;
#pragma omp parallel for simd	
for (_Atomic int j = 0; j < 64; j++)
;
#pragma omp for simd		
for (_Atomic int j = 0; j < 64; j++)
;
}
void
f2 (void)
{
_Atomic int i;
#pragma omp distribute		
for (i = 0; i < 64; i++)
;
#pragma omp distribute parallel for	
for (i = 0; i < 64; i++)
;
#pragma omp distribute parallel for simd 
for (i = 0; i < 64; i++)
;
#pragma omp distribute		
for (_Atomic int j = 0; j < 64; j++)
;
#pragma omp distribute parallel for	
for (_Atomic int j = 0; j < 64; j++)
;
#pragma omp distribute parallel for simd 
for (_Atomic int j = 0; j < 64; j++)
;
}
void
f3 (void)
{
int i;
_Atomic int j = 0;
#pragma omp simd linear(j:2)		
for (i = 0; i < 64; i++)
j += 2;
#pragma omp parallel for linear(j:1)	
for (i = 0; i < 64; i++)
j++;
}
