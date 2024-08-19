void
foo (int (&a)[8], int *b, int *c)
{
#pragma GCC unroll 8
for (int i : a)
a[i] = b[i] * c[i];
}
