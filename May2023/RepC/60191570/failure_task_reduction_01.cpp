void foo()
{
int x;
#pragma oss task weakreduction(+: x) reduction(+: x)
{
}
#pragma oss taskwait
}
