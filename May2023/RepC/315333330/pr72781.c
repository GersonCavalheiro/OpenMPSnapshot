int u;
void
foo (int *p)
{
int i;
#pragma omp for simd lastprivate(u) schedule (static, 32)	
for (i = 0; i < 1024; i++)
u = p[i];
}
void
bar (int *p)
{
int i;
#pragma omp taskloop simd lastprivate(u)	
for (i = 0; i < 1024; i++)
u = p[i];
}
