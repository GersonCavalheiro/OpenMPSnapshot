#include <omp.h>
#include <stdlib.h>
#include <stdio.h>
void main(void){
int b[3];
char* cptr;
int i;
cptr= malloc(sizeof(char));
#pragma omp parallel for
for( i= 0 ; i< 3 ; i++ ){
printf("%d\n", i);
}
}