void
foo (float n)
{
int A[n][n];	
#pragma omp parallel private(A)
;
}
