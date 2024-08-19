void
foo ()
{
#pragma omp parallel for
for (int i; i < 1; ++i)	
;
}
