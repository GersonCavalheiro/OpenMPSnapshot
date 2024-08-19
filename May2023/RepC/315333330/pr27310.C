struct A
{
~A ();
};
struct B
{
A a, b;
};
void
foo ()
{
A c, d;
#pragma omp parallel
B e;
}
