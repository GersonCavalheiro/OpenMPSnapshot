#include<assert.h>
struct C
{
int x;
int y[5];
};
#pragma omp task out(c->y[0:4])
void producer(struct C* c)
{
for (int i = 0; i < 5; ++i)
{
c->y[i] = 4;
}
}
#pragma omp task inout(c->y[0:4])
void consumer(struct C* c)
{
}
int main()
{
struct C c;
c.x = 1;
for (int i = 0; i < 5; ++i)
{
c.y[i] = -1;
}
producer(&c);
consumer(&c);
#pragma omp taskwait
for (int i = 0; i < 5; ++i)
{
c.y[i] = 4;
}
}
