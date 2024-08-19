int i;
template <int> void f1 ()
{
#pragma omp ordered
i++;
}
template <int> void f2 (bool p)
{
if (p)
{
#pragma omp master
i++;
}
}
void f3 ()
{
f1<0> ();
f2<0> (true);
}
