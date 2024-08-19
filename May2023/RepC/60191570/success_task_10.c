void foo(int*copy)
{
#pragma omp task inout([1]copy)
{
}
#pragma omp task inout(*copy)
{
}
}
int main()
{
int res = 0;
foo(&res);
#pragma omp taskwait
return 0;
}
