#include <stdio.h>
float x=0.0;
int y=0;
#pragma omp threadprivate(x,y)
int main (int argc, char * argv[])
{
#pragma omp parallel
{
#pragma omp single copyprivate(x,y)
{
x=1.0;
y=1;
}
}
printf ("x=%f y=%d\n", x, y);
return 0;
}
