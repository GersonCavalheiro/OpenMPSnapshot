#define N 1024
double a[N];
bool b[N];
char c[N];
void test ()
{
int i;
#pragma omp simd
for (i = 0; i < N; i++)
if ((c[i] > 0) && b[i])
a[i] = 0.0;
else
a[i] = 1.0;
}
