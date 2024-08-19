#include <stdio.h>
int main()
{
int x =0;
#pragma omp parallel for ordered 
for (int i = 0; i < 100; ++i) {
x++;
}
printf ("x=%d\n",x);
return 0;
} 
