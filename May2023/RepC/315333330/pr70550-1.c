#ifdef __SIZEOF_INT128__
typedef __int128 T;
#else
typedef long long T;
#endif
void bar (T);
#pragma omp declare target (bar)
void
foo (void)
{
{
int i;
#pragma omp target defaultmap(tofrom:scalar)	
{
i = 26;
bar (i);
}
}
{
T j;
#pragma omp target defaultmap(tofrom:scalar)	
{
j = 37;
bar (j);
}
}
{
int i;
#pragma omp target					
{
i = 26;
bar (i);
}
}
{
T j;
#pragma omp target					
{
j = 37;
bar (j);
}
}
{
int i;
#pragma omp target firstprivate (i)			
{
i = 26;
bar (i);
}
}
{
T j;
#pragma omp target firstprivate (j)			
{
j = 37;
bar (j);
}
}
{
int i;
#pragma omp target private (i)			
{
i = 26;
bar (i);
}
}
{
T j;
#pragma omp target private (j)			
{
j = 37;
bar (j);
}
}
}
