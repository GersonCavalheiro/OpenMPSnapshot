enum { N = 10 };
int main(int argc, char *argv[])
{
int v[N];
int *p = v;
#pragma omp task inout([N]p)
{}
return 0;
}
