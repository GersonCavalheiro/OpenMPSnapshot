#include<stdio.h>
#include<stdlib.h>
int* counter; 
int main()
{ 
counter = (int*) malloc(sizeof(int));
if (counter== NULL)
{
fprintf(stderr, "malloc() failes\n");
exit(1);
}
*counter = 0; 
#pragma omp parallel 
{
(*counter)++; 
}
printf("%d \n", *counter);
free (counter);
return 0;   
}
