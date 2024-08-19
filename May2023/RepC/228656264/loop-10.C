extern void abort (void);
int i = 8;
int main (void)
{
int j = 7, k = 0;
#pragma omp for
for (i = 0; i < 10; i++)
;
#pragma omp for
for (j = 0; j < 10; j++)
;
if (i != 8 || j != 7)
abort ();
#pragma omp parallel private (i) reduction (+:k)
{
i = 6;
#pragma omp for
for (i = 0; i < 10; i++)
;
k = (i != 6);
}
if (k)
abort ();
return 0;
}
