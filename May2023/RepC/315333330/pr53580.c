int
main ()
{
int x, y, v = 0;
#pragma omp parallel
#pragma omp for
for (x = 0; x < 10; x++)
{
#pragma omp for reduction(+: v)	
for (y = 0; y < 10; y++)
v++;
}
return v - 100;
}
