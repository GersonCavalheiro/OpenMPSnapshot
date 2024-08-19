#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
struct A
{
int x;
};
#pragma omp simd
void f(A& a, int i)
{
a.x = i;
}
void g()
{
int i;
#pragma omp simd
for (i = 0; i < 75; i++)
{
A a;
f(a, i);
}
}
int main(int argc, char *argv[])
{
g();
return 0;
}
