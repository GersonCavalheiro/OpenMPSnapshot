void bar (int);
void
foo (void)
{
int i, j, k, l, m, n, o, p, q;
#pragma omp task				
{
i = 2;
bar (i);
}
#pragma omp taskloop				
for (j = 0; j < 10; j++)
{
k = 7;
bar (k);
}
#pragma omp task firstprivate (l)		
{
l = 2;
bar (l);
}
#pragma omp taskloop firstprivate (m)		
for (j = 0; j < 10; j++)
{
m = 7;
bar (m);
}
#pragma omp task shared (n)			
{
n = 2;
bar (n);
}
#pragma omp taskloop shared (o)		
for (j = 0; j < 10; j++)
{
o = 7;
bar (o);
}
#pragma omp task private (p)			
{
p = 2;
bar (p);
}
#pragma omp taskloop shared (q)		
for (j = 0; j < 10; j++)
{
q = 7;
bar (q);
}
}
