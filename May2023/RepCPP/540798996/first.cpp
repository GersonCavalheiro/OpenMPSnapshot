#include <iostream>
#include <stdio.h>
#include <omp.h>
int main(){
#pragma omp parallel for
for(int i=0;i<100;i++){
printf("%i\n",i);
}
return 0;
}