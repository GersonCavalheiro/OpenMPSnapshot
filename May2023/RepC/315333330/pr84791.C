typedef int I;
template <int>
void
foo ()
{
I i;
#pragma omp parallel reduction (I::I: i)	
;						
}
template void foo<0> ();
