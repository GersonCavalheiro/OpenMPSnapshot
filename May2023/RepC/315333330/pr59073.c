void
foo ()
{
int i; 
#pragma omp distribute parallel for
for (i = 0; i < 10; i)	
;
}
