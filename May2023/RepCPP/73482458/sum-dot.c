#include <stdio.h>

#define NX 102400

int main(void)
{
double vecA[NX], vecB[NX], vecC[NX];


for (int i = 0; i < NX; i++) {
vecA[i] = 1.0 / ((double) (NX - i));
vecB[i] = vecA[i] * vecA[i];
}

double res = 0.0;

#pragma omp target data map(to:vecA, vecB) map(from:vecC, res)
{
#pragma omp target teams distribute parallel for
for (int i = 0; i < NX; i++) {
vecC[i] = vecA[i] + vecB[i];
}

#pragma omp target teams distribute parallel for reduction(+:res)
for (int i = 0; i < NX; i++) {
res += vecC[i] * vecB[i];
}

}

double sum = 0.0;

for (int i = 0; i < NX; i++) {
sum += vecC[i];
}
printf("Reduction sum: %18.16f\n", sum);
printf("Dot product: %18.16f\n", res);

return 0;
}
