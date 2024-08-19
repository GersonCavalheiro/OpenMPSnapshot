int
main ()
{
int y[8] = { 1, 2, 3, 4, 5, 6, 7, 8 };
auto x = [&y] ()
{
#pragma omp simd
for (int i = 0; i < 8; ++i)
y[i]++;
};
x ();
x ();
for (int i = 0; i < 8; ++i)
if (y[i] != i + 3)
__builtin_abort ();
}
