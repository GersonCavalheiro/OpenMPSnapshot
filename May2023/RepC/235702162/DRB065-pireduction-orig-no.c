#include <stdio.h>
int main(int argc, char * * argv)
{
double pi = 0.0;
long int i;
double x, interval_width;
int _ret_val_0;
interval_width=(1.0/((double)2000000000));
#pragma cetus private(i, x) 
#pragma loop name main#0 
#pragma cetus reduction(+: pi) 
#pragma cetus parallel 
#pragma omp parallel for private(i, x) reduction(+: pi)
for (i=0; i<2000000000; i ++ )
{
x=((i+0.5)*interval_width);
pi+=(1.0/((x*x)+1.0));
}
pi=((pi*4.0)*interval_width);
printf("PI=%f\n", pi);
_ret_val_0=0;
return _ret_val_0;
}
