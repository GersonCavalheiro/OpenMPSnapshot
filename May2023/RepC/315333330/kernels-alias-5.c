void
foo (int *a)
{
int *p = a;
#pragma acc kernels pcopyin (a[0:1], p[0:1])
{
*a = 0;
*p = 1;
}
}
