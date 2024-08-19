class A { A (const A&); };		
void foo (const A&);
void
bar (A& a)
{
#pragma omp task			
foo (a);
}
void
baz (A& a)
{
#pragma omp task firstprivate (a)	
foo (a);
}
