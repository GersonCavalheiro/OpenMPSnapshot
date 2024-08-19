int main()
{
#pragma omp parallel
{
#pragma omp single
{
int a;
#pragma omp task depend(inout : a)
{
}
}
}
}
