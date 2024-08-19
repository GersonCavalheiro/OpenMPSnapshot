#include <stdio.h>
#include <omp.h>
int main(){
int x = 0, y;
#pragma omp parallel num_threads(2)
{
int thrd = omp_get_thread_num();
if (thrd == 0) {
#pragma omp critical
{ x = 10; }
#pragma omp atomic write
y = 1;
} else {
int tmp = 0;
while (tmp == 0) {
#pragma omp atomic read acquire
tmp = y;
}
#pragma omp critical
{ if (x!=10) printf("x = %d\n", x); }
}
}
return 0;
}
