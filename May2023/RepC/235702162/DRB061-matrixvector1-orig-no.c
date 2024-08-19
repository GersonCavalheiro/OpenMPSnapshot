double a[100][100], v[100], v_out[100];
int init()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name init#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<100; i ++ )
{
#pragma cetus lastprivate(j) 
#pragma loop name init#0#0 
#pragma cetus parallel 
#pragma omp parallel for lastprivate(j)
for (j=0; j<100; j ++ )
{
a[i][j]=((i*j)+0.01);
}
v_out[i]=((i*j)+0.01);
v[i]=((i*j)+0.01);
}
_ret_val_0=0;
return _ret_val_0;
}
int mv()
{
int i, j;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name mv#0 
#pragma cetus parallel 
#pragma omp parallel for private(i, j)
for (i=0; i<100; i ++ )
{
double sum = 0.0;
#pragma cetus private(j) 
#pragma loop name mv#0#0 
#pragma cetus reduction(+: sum) 
#pragma cetus parallel 
#pragma omp parallel for private(j) reduction(+: sum)
for (j=0; j<100; j ++ )
{
sum+=(a[i][j]*v[j]);
}
v_out[i]=sum;
}
_ret_val_0=0;
return _ret_val_0;
}
int print()
{
int i, j, k;
int _ret_val_0;
#pragma cetus private(i, j) 
#pragma loop name print#0 
for (i=0; i<100; i ++ )
{
#pragma cetus private(j) 
#pragma loop name print#0#0 
for (j=0; j<100; j ++ )
{
printf("%lf\n", a[i][j]);
}
printf("%lf\n", v_out[i]);
printf("%lf\n", v[i]);
}
_ret_val_0=0;
return _ret_val_0;
}
int main()
{
int _ret_val_0;
init();
mv();
print();
_ret_val_0=0;
return _ret_val_0;
}
