#include <stdio.h>
#include <stdlib.h>
int foo(void)
{
return 999;
}
int main(int argc, char *argv[])
{
int result=666;
#pragma omp task out(result)
result = foo();
#pragma omp taskwait on(result)
if (result != 999) abort();
return 0;
}
