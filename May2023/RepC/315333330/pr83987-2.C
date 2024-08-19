struct A
{
int i;
};
struct B : virtual A
{
void foo();
};
void B::foo()
{
#pragma omp parallel
{
#pragma omp sections lastprivate (i)
{
i = 0;
}
}
}
