int n = 100, m = 100;
double b[100][100];
int init()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name init#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name init#0#0 
#pragma cetus parallel 
#pragma omp parallel for private(j)
for (j=0; j<m; j ++ )
{
b[i][j]=(i*j);
}
}
_ret_val_0=0;
return _ret_val_0;
}
void foo()
{
int i, j;
#pragma cetus private(i, j) 
#pragma loop name foo#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name foo#0#0 
for (j=0; j<(m-1); j ++ )
{
b[i][j]=b[i][j+1];
}
}
return ;
}
int print()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name print#0 
for (i=0; i<n; i ++ )
{
#pragma cetus private(j) 
#pragma loop name print#0#0 
for (j=0; j<m; j ++ )
{
printf("%lf\n", b[i][j]);
}
}
_ret_val_0=0;
return _ret_val_0;
}
int main()
{
int _ret_val_0;
init();
foo();
print();
_ret_val_0=0;
return _ret_val_0;
}
