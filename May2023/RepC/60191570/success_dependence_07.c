void foo(n)
{
int (*A)[n];
#pragma oss task in(A)
{
}
#pragma oss taskwait
}
