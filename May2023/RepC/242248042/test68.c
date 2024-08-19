#include<omp.h>
int main() {
volatile int sh;
#pragma omp parallel
{
long long i, j, k;
for(i = 0; i < 5000; i++)
for(j = 0; j < 100000; j++) {
#pragma omp flush
}
}
}
