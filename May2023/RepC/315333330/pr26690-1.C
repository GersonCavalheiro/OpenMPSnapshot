struct A			
{
A (int);				
};
void
foo ()
{
A a(0);
#pragma omp parallel private (a)	
;
}
