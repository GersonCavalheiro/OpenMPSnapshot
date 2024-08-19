#include <stdio.h>
#include <stdlib.h>
int main(int argc, char* argv[])
{
int i,j;
float temp, sum=0.0;
int len=100;
if (argc>1)
len = atoi(argv[1]);
float u[len][len];
for (i = 0; i < len; i++)
for (j = 0; j < len; j++)
u[i][j] = 0.5;
#pragma omp parallel for private (temp,i,j)
for (i = 0; i < len; i++)
for (j = 0; j < len; j++)
{
temp = u[i][j];
sum = sum + temp * temp;
}
printf ("sum = %f\n", sum); 
return 0;
}
