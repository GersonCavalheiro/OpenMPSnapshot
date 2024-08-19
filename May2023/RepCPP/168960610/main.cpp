#include <iostream>
#include <omp.h>

int main() 
{
printf("使用可能な最大スレッド数：%d\n", omp_get_max_threads());

#pragma omp parallel for
for (int i = 0; i < 10; i++) 
{
printf("thread = %d, i = %2d\n", omp_get_thread_num(), i);
}

return 0;
}