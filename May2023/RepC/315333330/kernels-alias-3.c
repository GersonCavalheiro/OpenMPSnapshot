void
foo (void)
{
unsigned int a;
unsigned int *p = &a;
#pragma acc kernels pcopyin (a, p[0:1])
{
a = 0;
*p = 1;
}
}
