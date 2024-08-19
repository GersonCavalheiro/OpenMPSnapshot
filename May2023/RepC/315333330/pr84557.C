template<int> struct A {};
template<int> struct B {};
void
foo ()
{
#pragma omp parallel firstprivate (A)		
;
#pragma omp parallel firstprivate (B<0>)	
;
}
