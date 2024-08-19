void foo(int n, ...);
int main()
{
int n = 3;
int x = 2, y = 2;
int empty_res;
int* res = &empty_res;
#pragma analysis_check assert upper_exposed(x, y, n, res) undefined(*res)
foo(n, x, y, res);
return 0;
}
