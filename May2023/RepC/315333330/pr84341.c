void
foo (int i)
{
#pragma omp atomic
i = &i + 1;		
}
