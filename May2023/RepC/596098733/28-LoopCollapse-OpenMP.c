#include <stdio.h>
#include <omp.h>
int WinMain() {
int i, j;
int n = 4;
int m = 3;
int a[n][m], b[n][m], c[n][m];
for (i = 0; i < n; i++) {
for (j = 0; j < m; j++) {
a[i][j] = i * j;
b[i][j] = i + j;
c[i][j] = 0;
}
}
#pragma omp parallel for collapse(2)
for (i = 0; i < n; i++) {
for (j = 0; j < m; j++) {
c[i][j] = a[i][j] + b[i][j];
}
}
for (i = 0; i < n; i++) {
for (j = 0; j < m; j++) {
printf("%d ", c[i][j]);
}
printf("\n");
}
return 0;
}
