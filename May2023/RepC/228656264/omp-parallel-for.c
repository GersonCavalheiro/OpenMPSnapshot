extern void abort (void);
int
main()
{
int i, a;
a = 30;
#pragma omp parallel for firstprivate (a) lastprivate (a) num_threads (2) schedule(static)
for (i = 0; i < 10; i++)
a = a + i;
if (a != 65)
abort ();
return 0;
}
