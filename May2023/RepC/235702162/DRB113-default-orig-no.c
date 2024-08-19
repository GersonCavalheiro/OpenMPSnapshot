int a[100][100];
int b[100][100];
int main()
{
int i, j;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name main#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<100; j ++ )
{
a[i][j]=i;
b[i][j]=i;
}
}
#pragma cetus private(i, j) 
#pragma loop name main#1 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#1#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<100; j ++ )
{
a[i][j]=(a[i][j]+1);
}
}
#pragma cetus private(i, j) 
#pragma loop name main#2 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#2#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<100; j ++ )
{
b[i][j]=(b[i][j]+1);
}
}
#pragma cetus private(i, j) 
#pragma loop name main#3 
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name main#3#0 
for (j=0; j<100; j ++ )
{
printf("%d %d\n", a[i][j], b[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
