#include <stdio.h>
#include <omp.h>
static float a[1000], b[1000], c[1000];
void test(int first, int last) 
{
int i;	
#pragma omp for schedule(static) ordered
for (i = first; i <= last; ++i) {
if (i % 2) 
{
#pragma omp ordered 
printf("test() iteration %d\n", i);
}
}
}
void test2(int iter) 
{
#pragma omp ordered
printf("test2() iteration %d\n", iter);
}
int main( ) 
{
int i;
#pragma omp parallel
{
test(1, 8);
#pragma omp for ordered
for (i = 0 ; i < 5 ; i++)
test2(i);
}
}
