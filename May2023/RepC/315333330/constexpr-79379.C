template <int N>
constexpr int
foo (int x)
{
int q[64] = { 0 }, r = 0;
#pragma GCC ivdep
for (int i = 0; i < x; ++i)
q[i] += 2;
for (int i = 0; i < x; ++i)
r += q[i];
return r + N;
}
constexpr int a = foo<0> (17);
static_assert (a == 34, "");
