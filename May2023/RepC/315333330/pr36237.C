struct A
{
~A ();
};
struct B
{
B (const A &x = A ()) : a (x) { }
A a;
};
B var;
void bar ();
void
foo ()
{
#pragma omp parallel private (var)
bar ();
}
