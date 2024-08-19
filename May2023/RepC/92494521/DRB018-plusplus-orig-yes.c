#include <stdlib.h>
#include <stdio.h>
int input[1000]; 
int output[1000];
int main()
{
int i ;
int inLen=1000 ; 
int outLen = 0;
for (i=0; i<inLen; ++i) 
input[i]= i;  
#pragma omp parallel for
for (i=0; i<inLen; ++i) 
{
output[outLen++] = input[i] ;
}  
printf("output[500]=%d\n",output[500]);
return 0;
}
