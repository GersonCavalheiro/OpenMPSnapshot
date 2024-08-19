#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int work1()
{
int j, tid;
tid = omp_get_thread_num();
for (j = 0; j < 10; j++)
printf("The value of j as printed by work 1, thread %li = %li\n", tid, j);
}
int work2()
{
int j, tid;
tid = omp_get_thread_num();
for (j = 0; j < 10; j++)
printf("The value of j as printed by work 2, thread %li = %li\n", tid, j);
}
int work3()
{
printf("Work 3\n");
printf("Work 3\n");
printf("Work 3\n");
printf("Work 3\n");
printf("Work 3\n");
}
int work4()
{
printf("Work 4\n");
printf("Work 4\n");
printf("Work 4\n");
printf("Work 4\n");
printf("Work 4\n");
}

main()
{
#pragma omp parallel sections
{
work1();
#pragma omp section
{
work2();
work3();
}
#pragma omp section
{
work4();
}
}
}
