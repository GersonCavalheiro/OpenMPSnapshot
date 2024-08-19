struct A
{
~A () {}
};
struct B
{
A a;
B ();
};
void
foo ()
{
#pragma omp parallel
{
B b[1];
new int;
}
}
