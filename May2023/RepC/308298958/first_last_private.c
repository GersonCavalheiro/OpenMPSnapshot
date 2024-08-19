#include <omp.h>
#include <stdio.h>

int main(){
int i, x;


x = 5;
printf("before parallelism, x: %d\n\n", x);


printf("private(x):\n");
#pragma omp parallel for private(i, x)
for(i = 0; i < 10; i++){
if(i == 0){
printf("thread %d, before tweak, x: %d\n", omp_get_thread_num(), x);
x = 0;
printf("thread %d, after tweak, x: %d\n\n", omp_get_thread_num(), x);
}
}

printf("after private(x), x: %d\n\n", x);


printf("firstprivate(x):\n");
#pragma omp parallel for private(i) firstprivate(x)
for(i = 0; i < 10; i++){
if(i == 0){
printf("thread %d, before tweak, x: %d\n", omp_get_thread_num(), x);
x = 0;
printf("thread %d, after tweak, x: %d\n\n", omp_get_thread_num(), x);
}
}

printf("after firstprivate(x), x: %d\n\n", x);


printf("lastprivate(x):\n");
#pragma omp parallel for private(i) lastprivate(x)
for(i = 0; i < 10; i++){
if(i == 9){
printf("thread %d, before tweak, x: %d\n", omp_get_thread_num(), x);
x = 0;
printf("thread %d, after tweak, x: %d\n\n", omp_get_thread_num(), x);
}
}

printf("after lastprivate(x), x: %d\n", x);

return 0;
}
