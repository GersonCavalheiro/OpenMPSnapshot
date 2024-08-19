


#include <stdlib.h>
int main(int argc, char* argv[]) 
{
int i,j;
int len=100;
if (argc>1)
len = atoi(argv[1]);

int n=len, m=len;
double b[n][m];
#pragma omp parallel for private(j)
for (i=1;i<n;i++)
for (j=0;j<m;j++) 
b[i][j]=b[i][j-1];

return 0;     
}
