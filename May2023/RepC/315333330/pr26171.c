int thrv = 0;
#pragma omp threadprivate (thrv)
int
main ()
{
thrv = 1;
return 0;
}
