#define SIZE 10240
float a[SIZE] __attribute__((__aligned__(32)));
float b[SIZE] __attribute__((__aligned__(32)));
float c[SIZE] __attribute__((__aligned__(32)));
#pragma GCC push_options
#pragma GCC optimize (3, "unroll-all-loops", "-fprefetch-loop-arrays")
void
opt3 (void)
{
int i;
for (i = 0; i < SIZE; i++)
a[i] = b[i] + c[i];
}
#pragma GCC pop_options
void
not_opt3 (void)
{
int i;
for (i = 0; i < SIZE; i++)
a[i] = b[i] - c[i];
}
