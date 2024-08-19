#include<assert.h>
int main() {
int j = 0;
#pragma omp taskloop grainsize(100)
for (int i = 0; i <= j; ++i)
{
assert(i == 0);
}
#pragma omp taskloop num_tasks(100)
for (int i = 0; i <= j; ++i)
{
assert(i == 0);
}
#pragma omp taskloop grainsize(100)
for (int i = 0; i < j; ++i)
{
assert(0);
}
#pragma omp taskloop num_tasks(100)
for (int i = 0; i < j; ++i)
{
assert(0);
}
}
