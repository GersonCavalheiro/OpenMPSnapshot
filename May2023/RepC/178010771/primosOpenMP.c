#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>


int main(int argc, char const *argv[])
{
int div=1,i,j,n;

if(argc >= 2){
n = atoi(argv[1]); 
}else{
n = 262144;
}

printf("[");

#pragma omp parallel shared(n) private(i,j)
#pragma omp for schedule(static)
for(i = 1; i <= n; i++){
div=1;
for(j = 2; j < i; j++){
if(i % j == 0){
div=0;
}
}

if(div){
printf("%i ", i);
}
}
printf("]\n");

return 0;
}
