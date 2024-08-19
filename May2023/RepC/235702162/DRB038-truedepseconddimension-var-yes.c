#include <stdlib.h>
int main(int argc, char * argv[])
{
int i, j;
int len = 1000;
int n = len, m = len;
double b[n][m];
int _ret_val_0;
if (argc>1)
{
len=atoi(argv[1]);
}
#pragma cetus private(i, j) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<m; j ++ )
{
b[i][j]=(i+j);
}
}
#pragma cetus private(i, j) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#1#0 
for (j=1; j<m; j ++ )
{
b[i][j]=b[i][j-1];
}
}
#pragma cetus private(i, j) 
#pragma loop name main#2 
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#2#0 
for (j=0; j<m; j ++ )
{
printf("%lf\n", b[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
