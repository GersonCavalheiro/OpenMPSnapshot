struct A;
void
foo ()
{
#pragma omp parallel private(A)	
;
}
