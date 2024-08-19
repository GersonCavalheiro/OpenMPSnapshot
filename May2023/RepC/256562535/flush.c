#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
int main(void) {
int data, flag = 0;
#pragma omp parallel num_threads(2)
{
if (omp_get_thread_num() == 0) {
data = 42;
#pragma omp flush(flag, data)
flag = 1;
#pragma omp flush(flag)
} else if (omp_get_thread_num() == 1) {
#pragma omp flush(flag, data)
while (flag < 1) {
#pragma omp flush(flag, data)
}
printf("flag=%d data=%d\n", flag, data);
#pragma omp flush(flag, data)
printf("flag=%d data=%d\n", flag, data);
}
}
return 0;
}
