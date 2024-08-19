#include <stdlib.h>
int main(int argc, char* argv[])
{
int i,j;
int len = 1000;
if (argc>1)
len = atoi(argv[1]);
int n=len, m=len;
double b[len][len];
for (i=0; i<n; i++)
for (j=0; j<m; j++)
b[i][j] = 0.5; 
#pragma omp parallel for private(j)
for (i=1;i<n;i++)
for (j=1;j<m;j++)
b[i][j]=b[i-1][j-1];
return 0;
}
