struct S
{
S () {}
S (S const &) {}
};
struct T
{
S s;
};
void
bar (T &)
{
}
void
foo ()
{
T t;
#pragma omp task
bar (t);
}
