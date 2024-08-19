typedef struct {
char stuff[400];
} foo;
void
funk (foo *begin, foo *end)
{
foo *p;
#pragma omp parallel for ordered(1)
for (p=end; p > begin; p--)
{
#pragma omp ordered depend(sink:p+1)
void bar ();
bar();
#pragma omp ordered depend(source)
}
}
