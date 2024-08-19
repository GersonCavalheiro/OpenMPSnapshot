struct A { A(); ~A(); int i; };
int
foo ()
{
A a;
#pragma omp parallel private (a)
for (int i = 0; i < 5; ++i)
a.i++;
return 0;
}
int
bar ()
{
A a;
#pragma omp for private (a)
for (int i = 0; i < 5; ++i)
a.i++;
return 0;
}
