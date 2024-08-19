#include <omp.h>
#include <stdio.h>
#include <unistd.h>
int main(){
int x = 0;
int i;
#pragma omp parallel num_threads(8)
{
if(omp_get_thread_num() == 0){
for(i = 1; i <= 10; i++){
x += i;
#pragma omp flush
}
}
else{
sleep(1);
printf("x : %d\n", x);
}
}
return 0;
}
