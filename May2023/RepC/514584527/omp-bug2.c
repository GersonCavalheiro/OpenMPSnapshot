#include <stdio.h>
#include <omp.h>
int main( void )
{
const int foo = 1;
#pragma omp parallel default(none) shared(foo)
{
int baz = 0;
baz += foo;
printf("Thread %d: baz=%d\n", omp_get_thread_num(), baz);
}
return 0;
}
