#include <stdio.h>
int main( void )
{
int a = 2;
#pragma omp parallel reduction(*:a)
{
a += 2;
}
printf("%d\n",a);
return 0;
}
