extern void bar (int);
int j;
void test (void)
{
#pragma GCC unroll 0
#pragma GCC ivdep
for (unsigned long i = 1; i <= 3; ++i)
bar(i);
#pragma GCC ivdep
#pragma GCC unroll 0
for (unsigned long i = 1; i <= j; ++i)
bar(i);
}
