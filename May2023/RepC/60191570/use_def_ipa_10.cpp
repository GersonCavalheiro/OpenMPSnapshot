int N = 0;
int M = 1;
void foo(int *p, int *&q, int u, int &v)
{
p = &N;
*p = 5;
q = &M;
*q = 10;
u = v + 2;
v = u;
}
int main()
{
int *a, *b;
int c, d;
#pragma analysis_check assert upper_exposed(a, b, c, d) defined(*a, b, *b, d)
foo(a, b, c, d);
return 0;
}