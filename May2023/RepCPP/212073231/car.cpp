#include <iostream>

int car() {
#pragma omp parallel num_threads(3) default(none)
{
#pragma omp single
{
printf("A ");
#pragma omp task default(none)
{
printf("race ");
}
#pragma omp task default(none)
{
printf("car ");
}
#pragma omp taskwait
printf("is fun to watch ");
}
} 

printf("\n");
return (0);
}