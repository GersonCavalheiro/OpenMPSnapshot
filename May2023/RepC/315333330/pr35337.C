struct A { };
void
foo ()
{
#pragma omp parallel firstprivate(A)	
;
}
void
bar ()
{
#pragma omp for lastprivate(A)		
for (int i = 0; i < 10; i++)
;
}
