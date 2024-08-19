static int N = 0;
int M = 0;
void foo1(int n = N, int* m = &M);
void foo2(int n = N, int* m = &M)
{
n += 1;
*m += 1;
}
int main()
{
#pragma analysis_check assert undefined(N, M)
foo1();
#pragma analysis_check assert upper_exposed(N, M) defined(M)
foo2();
return 0;
}
