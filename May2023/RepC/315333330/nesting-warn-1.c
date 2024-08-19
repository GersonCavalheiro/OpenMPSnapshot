extern int i;
void
f_omp_target (void)
{
#pragma omp target
{
#pragma omp target 
;
#pragma omp target data map(i) 
;
#pragma omp target update to(i) 
#pragma omp parallel
{
#pragma omp target 
;
#pragma omp target data map(i) 
;
#pragma omp target update to(i) 
}
}
}
