#include <string.h>
int main(int argc,char *argv[])
{
int i;
int j;
double a[20][20];
memset(a,0,(sizeof(a)));
for (i = 0; i < 20 -1; i += 1) {
#pragma omp parallel for
for (j = 0; j < 20; j += 1) {
a[i][j] += a[i + 1][j];
}
}
return 0;
}
