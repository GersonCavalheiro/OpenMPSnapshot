void
a20_wrong ()
{
int a = 1;
#pragma omp parallel
{
if (a != 0)
#pragma omp flush(a)	
if (a != 0)
#pragma omp barrier	
}
}
