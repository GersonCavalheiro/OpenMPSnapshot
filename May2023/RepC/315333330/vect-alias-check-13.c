void
f1 (int *x, long step1, int n)
{
for (int i = 0; i < n; ++i)
x[i * step1] += 1;
}
void
f2 (int *x, long step2, int n)
{
#pragma GCC ivdep
for (int i = 0; i < n; ++i)
x[i * step2] += 2;
}
