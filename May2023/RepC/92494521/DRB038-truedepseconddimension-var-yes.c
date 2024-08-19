#include <stdlib.h>
int main(int argc, char* argv[]) 
{
int i,j;
int len = 1000;
if (argc>1)
len = atoi(argv[1]);
int n=len, m=len;
double b[n][m];
for (i=0;i<n;i++)
#pragma omp parallel for
for (j=1;j<m;j++)
b[i][j]=b[i][j-1];
return 0;
}
