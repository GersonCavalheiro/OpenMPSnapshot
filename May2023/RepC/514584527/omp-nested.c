#include <stdio.h>
#include <omp.h>
void greet(int level, int parent)
{
printf("Level %d (parent=%d): greetings from thread %d of %d\n", 
level, parent, omp_get_thread_num(), omp_get_num_threads());
}
int main( void )
{
omp_set_num_threads(4);
#pragma omp parallel
{
greet(1, -1);
int parent = omp_get_thread_num();
#pragma omp parallel
{
greet(2, parent);
int parent = omp_get_thread_num();
#pragma omp parallel
{
greet(3, parent);
}
}
}    
return 0;
}
