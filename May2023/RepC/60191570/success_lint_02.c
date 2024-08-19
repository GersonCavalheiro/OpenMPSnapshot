int main()
{
int *p = 0;
int N = 100;
#pragma oss lint alloc(p[0;N])
{
}
#pragma oss lint free(*p)
{
}
}
