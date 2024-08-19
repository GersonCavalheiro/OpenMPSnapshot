#include<stdio.h>
void f1(int* q)
{
*q += 1;
}
int main()
{ 
int i=0; 
#pragma omp parallel 
{
f1(&i);
}
printf ("i=%d\n",i);
return 0;   
}
