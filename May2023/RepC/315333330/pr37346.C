struct A
{
A ();
int i;
};
A::A ()
{
#pragma omp critical
i++;
}
