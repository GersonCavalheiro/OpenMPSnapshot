struct A
{
A (int x = 6);			
A (long long x = 12LL);		
};
void
foo ()
{
A a(6);
#pragma omp parallel private (a)	
;
}
