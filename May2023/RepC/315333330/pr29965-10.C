void
foo1 ()
{
int i;
#pragma omp for schedule (static)
for (i = 0; i < 2834; i++)
throw 0;
}
void
foo2 ()
{
int i;
#pragma omp parallel for schedule (static)
for (i = 0; i < 2834; i++)
throw 0;
}
