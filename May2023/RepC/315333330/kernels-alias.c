#define N 2
void
foo (void)
{
unsigned int a[N];
unsigned int b[N];
unsigned int c[N];
unsigned int d[N];
#pragma acc kernels copyin (a) create (b) copyout (c) copy (d)
{
a[0] = 0;
b[0] = 0;
c[0] = 0;
d[0] = 0;
}
}
