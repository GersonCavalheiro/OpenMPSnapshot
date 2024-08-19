const int N = 100;
int main(int argc, char** argv)
{
int v;
int *ptr;
int a[N];
#pragma analysis_check assert upper_exposed() defined(ptr, *ptr)
{
ptr = &v;
*ptr = 10;
}
#pragma analysis_check assert upper_exposed() defined(ptr)
ptr = &a[0];
#pragma analysis_check assert upper_exposed(a) defined(ptr)
ptr = a;
#pragma analysis_check assert upper_exposed(argc) defined(ptr)
ptr = &a[argc];
return 0;
}
