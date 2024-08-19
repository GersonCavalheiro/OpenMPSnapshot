#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#pragma omp simd
void f(int& a, int i)
{
a = i;
}
void g()
{
int i;
#pragma omp simd
for (i = 0; i < 75; i++)
{
int a;
f(a, i);
}
}
int main(int, char**)
{
g();
}
