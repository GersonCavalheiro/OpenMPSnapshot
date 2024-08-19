struct A
{
operator int () const;
A& operator += (int);
A& operator ++ ();
};
void
foo (A a, A b)
{
#pragma omp for
for (A i = a; i <=; ++i)	
;				
}
