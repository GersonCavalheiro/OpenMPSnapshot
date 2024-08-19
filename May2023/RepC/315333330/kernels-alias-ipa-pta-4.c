void
foo (void)
{
unsigned int a;
unsigned int b;
unsigned int c;
#pragma acc kernels pcopyout (a, b, c)
{
a = 0;
b = 1;
c = a;
}
}
