struct S;			
extern __thread struct S s;	
struct T;
extern __thread struct T t;
void
foo ()
{
#pragma omp parallel copyin (s)
;
}
