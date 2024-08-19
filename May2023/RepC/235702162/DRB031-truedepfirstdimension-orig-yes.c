#include <stdlib.h>
#include <stdio.h>
int main(int argc, char * argv[])
{
int i, j;
int n = 1000, m = 1000;
double b[1000][1000];
int _ret_val_0;
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
b[i][j]=0.5;
}
}
#pragma cetus private(i, j) 
#pragma loop name main#1 
for (i=1; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#1#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=1; j<m; j ++ )
{
b[i][j]=b[i-1][j-1];
}
}
printf("b[500][500]=%f\n", b[500][500]);
_ret_val_0=0;
return _ret_val_0;
}
