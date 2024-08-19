#define REPEAT8(X, P) \
X (P##0) X (P##1) X (P##2) X (P##3) X (P##4) X (P##5) X (P##6) X (P##7)
#define REPEAT40(X) \
REPEAT8 (X, 0) REPEAT8 (X, 1)  REPEAT8 (X, 2) REPEAT8 (X, 3) REPEAT8 (X, 4)
volatile int testi;
void __attribute__ ((weak))
f1 (int x[40][100], int *y)
{
asm volatile ("" ::: "memory");
if (y[testi] == x[testi][testi])
throw 100;
}
void __attribute__ ((weak))
f2 (int x[40][100], int *y)
{
#define DECLARE(N) int y##N = y[N];
REPEAT40 (DECLARE);
for (int j = 0; j < 20; ++j)
{
f1 (x, y);
#pragma omp simd
for (int i = 0; i < 100; ++i)
{
#define INC(N) x[N][i] += y##N;
REPEAT40 (INC);
}
}
}
void __attribute__ ((weak))
f3 (int x[40][100], int *y, int *z)
{
volatile int extra = 111;
try
{
f2 (x, y);
}
catch (int val)
{
*z = val + extra;
}
}
static int x[40][100];
static int y[40];
static int z;
int
main (void)
{
f3 (x, y, &z);
if (z != 211)
__builtin_abort ();
return 0;
}
