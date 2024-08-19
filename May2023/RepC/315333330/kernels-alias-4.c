#define N 2
void
foo (void)
{
unsigned int a[N];
unsigned int *p = &a[0];
#pragma acc kernels pcopyin (a, p[0:2])
{
a[0] = 0;
*p = 1;
}
}
