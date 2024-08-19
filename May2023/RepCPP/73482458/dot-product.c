#include <stdio.h>

#define NX 102400

int main(void)
{
double vecA[NX], vecB[NX];


for (int i = 0; i < NX; i++) {
vecA[i] = 1.0 / ((double) (NX - i));
vecB[i] = vecA[i] * vecA[i];
}

double res = 0.0;
#pragma omp target teams distribute parallel for map(to:vecA,vecB) reduction(+:res)
for (int i = 0; i < NX; i++) {
res += vecA[i] * vecB[i];
}

printf("Dot product: %18.16f\n", res);

return 0;
}
