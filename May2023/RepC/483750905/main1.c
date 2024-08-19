#include <stdio.h>;
#include "omp.h";
void main()
{
#pragma omp parallel	
{
int ID = 0;
printf("hello(%d)",ID);
printf("word(%d) \n",ID);
}
}
