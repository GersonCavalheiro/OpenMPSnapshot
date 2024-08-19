#include<omp.h>
#include<assert.h>
const int CHUNKSIZE = 2;
int main() {
int num_its_executed_in_a_chunk = 0;
#pragma omp for schedule(dynamic, CHUNKSIZE) firstprivate(num_its_executed_in_a_chunk)
for (int i = 100; i >= 0; i-=2)
{
num_its_executed_in_a_chunk++;
assert(num_its_executed_in_a_chunk <= CHUNKSIZE);
}
}
