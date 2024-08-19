template <typename T>
void
foo (T (&a)[8], T *b, T *c)
{
#pragma GCC unroll 8
for (int i : a)
a[i] = b[i] * c[i];
}
void
bar (int (&a)[8], int *b, int *c)
{
foo <int> (a, b, c);
}
