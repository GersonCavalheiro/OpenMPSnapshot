struct S { int a; };
void
foo (struct S *x)
{
struct S b;
#pragma omp parallel private (b.a)	
;
#pragma omp parallel private (x->a)	
;
}
