int main()
{
int i, j;
int n = 100, m = 100;
double b[n][m];
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
for (j=0; j<n; j ++ )
{
b[i][j]=((double)(i*j));
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
#pragma cetus private(i, j) 
#pragma loop name main#2 
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#2#0 
for (j=0; j<n; j ++ )
{
printf("%lf\n", b[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
