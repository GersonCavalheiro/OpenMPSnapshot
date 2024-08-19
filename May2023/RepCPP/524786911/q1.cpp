#include <iostream>
#include <omp.h>
using namespace std;
#define N 1024

int main() {
int i, k = 10;
int a[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
int c[1000];
int b[N][N];
int loc = -1;
int tmp = -1;

#pragma omp parallel for schedule(static, 1)
for(i = 0; i < k; i++)             
b[i][k] = b[a[i]][k];

printf("%d %d", a[0], b[0][0]);

for(i = 0; i < 1000; i++) {
tmp = tmp + 1;
c[i] = tmp;
}

#pragma omp parallel for schedule(static, 1)
for(i = 0; i < 1000; i++) {
if (c[i] % 4 == 0) {
#pragma omp critical
{
if (i > loc)
loc = i;
}
}
}
return 0;
}