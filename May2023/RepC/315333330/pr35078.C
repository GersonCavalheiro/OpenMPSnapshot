template<int> void
foo ()
{
#pragma omp parallel for
for (int& i = 0; i < 10; ++i)	
;
}
void
bar ()
{
int j = 0;
#pragma omp parallel for
for (int& i = j; i < 10; ++i)	
;
}
