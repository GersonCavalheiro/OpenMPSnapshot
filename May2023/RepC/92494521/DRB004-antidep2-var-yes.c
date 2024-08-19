#include <stdlib.h>
int main(int argc,char *argv[])
{
int i, j;
int len = 20; 
if (argc>1)
len = atoi(argv[1]);
double a[len][len];
for (i=0; i< len; i++)
for (j=0; j<len; j++)
a[i][j] = 0.5; 
#pragma omp parallel for private(j)
for (i = 0; i < len - 1; i += 1) {
for (j = 0; j < len ; j += 1) {
a[i][j] += a[i + 1][j];
}
}
return 0;
}
