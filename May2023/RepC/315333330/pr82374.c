#define SIZE (1024 * 1024)
float a[SIZE];
float b[SIZE];
float c[SIZE];
float d[SIZE];
__attribute__((optimize ("O2", "tree-vectorize"))) void
foo (void)
{
int i;
#pragma omp parallel for
for (i = 0; i < SIZE; i++)
c[i] = a[i] + b[i];
}
__attribute__((optimize ("O2", "tree-vectorize"))) void
bar (void)
{
int i;
for (i = 0; i < SIZE; i++)
d[i] = a[i] + b[i];
}
