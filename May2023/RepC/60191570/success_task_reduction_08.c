#include <stdbool.h>
#include <assert.h>
#define N 10
#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
int main()
{
int x;
x = 0;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(+: x)
{
x += 2;
}
}
#pragma omp task reduction(+: x)
{
}
#pragma omp taskwait
assert(x == N*2);
x = 10;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(*: x)
{
x *= 2;
}
}
#pragma omp task reduction(*: x)
{
}
#pragma omp taskwait
assert(x == 10*(1 << N));
x = 100;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(-: x)
{
x -= 2;
}
}
#pragma omp task reduction(-: x)
{
}
#pragma omp taskwait
assert(x == 100 - N*2);
x = ~0;
for (int i = 0; i < sizeof(int)*8; ++i)
{
#pragma omp task reduction(&: x) firstprivate(i)
{
if (i%2 == 0)
x &= ~(1 << i);
}
}
#pragma omp task reduction(&: x)
{
}
#pragma omp taskwait
for (int j = 0; j < sizeof(int); ++j)
{
assert(((unsigned char*)&x)[j] == 0xAA);
}
x = 0;
for (int i = 0; i < sizeof(int)*8; ++i)
{
#pragma omp task reduction(|: x) firstprivate(i)
{
if (i%2 == 0)
x |= (1 << i);
}
}
#pragma omp task reduction(|: x)
{
}
#pragma omp taskwait
for (int j = 0; j < sizeof(int); ++j)
{
assert(((unsigned char*)&x)[j] == 0x55);
}
x = ~0;
for (int i = 0; i < sizeof(int)*8; ++i)
{
#pragma omp task reduction(^: x) firstprivate(i)
{
if (i%2 == 0)
x ^= (1 << i);
}
}
#pragma omp task reduction(^: x)
{
}
#pragma omp taskwait
for (int j = 0; j < sizeof(int); ++j)
{
assert(((unsigned char*)&x)[j] == 0xAA);
}
x = true;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(&&: x) firstprivate(i)
{
x = x && true;
}
}
#pragma omp task reduction(&&: x)
{
}
#pragma omp taskwait
assert(x);
x = false;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(||: x) firstprivate(i)
{
if (i%2 == 0)
x = x || true;
else
x = x || false;
}
}
#pragma omp task reduction(||: x)
{
}
#pragma omp taskwait
assert(x);
x = 0;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(max: x) firstprivate(i)
{
x = MAX(x, i);
}
}
#pragma omp task reduction(max: x)
{
}
#pragma omp taskwait
assert(x == N - 1);
x = N;
for (int i = 0; i < N; ++i)
{
#pragma omp task reduction(min: x) firstprivate(i)
{
x = MIN(x, i);
}
}
#pragma omp task reduction(min: x)
{
}
#pragma omp taskwait
assert(x == 0);
}
