#include <stdio.h>
int main( void )
{
int a=1, b=1, c=1, d=1;	
#pragma omp parallel num_threads(10) private(a) shared(b) firstprivate(c)
{	
printf("Hello World!\n");
a++;	
b++;	
c++;	
d++;	
}	
printf("a=%d\n", a);
printf("b=%d\n", b);
printf("c=%d\n", c);
printf("d=%d\n", d);
return 0;
}
