void
foo (void)
{
unsigned int a;
unsigned int b;
unsigned int c;
unsigned int d;
#pragma acc kernels copyin (a) create (b) copyout (c) copy (d)
{
a = 0;
b = 0;
c = 0;
d = 0;
}
}
