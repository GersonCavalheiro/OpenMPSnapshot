void foo()
{
int x = 0;
#pragma oss task reduction(+: x)
{
#pragma oss release reduction(+: x)
{
}
}
#pragma oss task weakreduction(+: x)
{
#pragma oss release weakreduction(+: x)
{
}
}
#pragma oss taskwait
}
