int
main (void)
{
#pragma omp parallel
{
#pragma omp for schedule (guided, 10)
for (int i = 0; i < 1826; i += 10)
;
#pragma omp for schedule (guided, 10)
for (int i = 0; i > -1826; i -= 10)
;
}
return 0;
}
