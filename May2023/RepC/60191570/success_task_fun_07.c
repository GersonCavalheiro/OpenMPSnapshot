#include<assert.h>
#define N 5
const int M = N;
int array[N];
#pragma omp task inout(a[0:4])
void foo(int a[])
{
int i = 0;
for (i = 0; i < M; i++, a++)
{
(*a) = i;
}
}
int main()
{
int* first_position_array = array;
foo(array);
#pragma omp taskwait
int i = 0;
for (i = 0; i < M; i++, first_position_array++)
{
assert((*first_position_array) == i);
}
}
