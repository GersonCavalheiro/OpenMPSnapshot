#include <stdlib.h>
#include <stdio.h>
int main(int argc, char* argv[])
{
int i ;
int inLen=1000 ; 
int outLen = 0;
if (argc>1)
inLen= atoi(argv[1]);
int input[inLen]; 
int output[inLen];
for (i=0; i<inLen; ++i) 
input[i]=i; 
#pragma omp parallel for
for (i=0; i<inLen; ++i) {
output[outLen++] = input[i] ;
}  
printf("output[0]=%d\n", output[0]);
return 0;
}
