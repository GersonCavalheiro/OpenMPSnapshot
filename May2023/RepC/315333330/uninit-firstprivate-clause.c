void
foo (void)
{
int i;
#pragma acc parallel
{
i = 1;
}
}
void
foo2 (void)
{
int i;
#pragma acc parallel firstprivate (i) 
{
i = 1;
}
}
