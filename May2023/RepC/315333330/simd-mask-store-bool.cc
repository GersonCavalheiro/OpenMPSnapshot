#define N 1024
int a[N], b[N], c[N];
bool d[N];
void
test (void)
{
int i;
#pragma omp simd safelen(64)
for (i = 0; i < N; i++)
if (a[i] > 0)
d[i] = b[i] > c[i];
}
