#include<assert.h>
#pragma omp task out(c[4][0:4])
void producer(int c[][5])
{
int i;
for (i = 0; i < 5; ++i)
{
c[4][i] = 4;
}
}
#pragma omp task inout(c[4][0:4])
void consumer(int c[][5])
{
}
int main()
{
int c[5][5] = { 0 };
int i;
producer(c);
consumer(c);
#pragma omp taskwait
for (i = 0; i < 5; ++i)
{
assert(c[4][i] == 4);
}
}
