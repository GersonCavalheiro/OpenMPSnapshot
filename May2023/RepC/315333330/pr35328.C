struct A
{
~A ()();		
};
struct B
{
B ()();		
};
struct C
{
C ();
C (const C &)();	
};
void
foo ()
{
A a;
B b;
C c;
#pragma omp parallel firstprivate (a)
;
#pragma omp parallel private (b)
;
#pragma omp parallel firstprivate (c)
;
}
