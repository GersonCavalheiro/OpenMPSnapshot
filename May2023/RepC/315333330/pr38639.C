template<int> void
foo ()
{
#pragma omp parallel for
for (auto i = i = 0; i<4; ++i)	
;
}
void
bar ()
{
foo<0> ();
}
