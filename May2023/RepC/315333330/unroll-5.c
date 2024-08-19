extern void bar (int);
int j;
void test (void)
{
#pragma GCC unroll 4+4
for (unsigned long i = 1; i <= 8; ++i)
bar(i);
#pragma GCC unroll -1	
for (unsigned long i = 1; i <= 8; ++i)
bar(i);
#pragma GCC unroll 20000000000	
for (unsigned long i = 1; i <= 8; ++i)
bar(i);
#pragma GCC unroll j	
for (unsigned long i = 1; i <= 8; ++i)
bar(i);
#pragma GCC unroll  4.2	
for (unsigned long i = 1; i <= 8; ++i)
bar(i);
}
