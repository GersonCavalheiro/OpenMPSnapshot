int x, *p = &x;
extern void abort (void);
void
f1 (int *q)
{
*q = 1;
#pragma omp flush
}
void
f2 (int *q)
{
#pragma omp barrier
*q = 2;
#pragma omp barrier
}
int
g (int n)
{
int i = 1, j, sum = 0;
*p = 1;
#pragma omp parallel reduction(+: sum) num_threads(2)
{
f1 (&j);
sum += j;
f2 (&j);
sum += i + j + *p + n;
}
return sum;
}
int
main ()
{
int result = g (10);
if (result != 30)
abort ();
return 0;
}
