int main()
{
int ok;
ok = foo();
return ok;
}
int foo()
{
#pragma omp task
{
}
#pragma omp taskwait
return 0;
}
