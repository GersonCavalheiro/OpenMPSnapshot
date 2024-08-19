#include <assert.h>
#include <stdint.h>
#include <string.h>
#pragma omp declare reduction(my_add: int : omp_out = omp_in + omp_out) initializer(omp_priv = omp_orig)
int main(int argc, char *argv[]) {
int sum[10];
memset(sum, 1, sizeof(sum));
#pragma omp parallel
{
#pragma omp for reduction(my_add : sum)
for (int32_t i = 0; i < 10; i++) sum[i] += i;
}
for (int32_t i = 0; i < 10; i++)
assert(sum[i] == i + 1);
}
