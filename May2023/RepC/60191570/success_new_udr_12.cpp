#include <vector>
int bar( int * a, const int * b)
{
*a = *b + 1;
return *a;
}
#pragma omp declare reduction( foo : int : bar (&omp_out, &omp_in) )
int main (int argc, char* argv[])
{
std::vector<int> v(10,1);
int x;
#pragma omp parallel for reduction (foo : x)
for (int i=0; i<10; i++)
{
x += v[i];
}
return 0;
}
