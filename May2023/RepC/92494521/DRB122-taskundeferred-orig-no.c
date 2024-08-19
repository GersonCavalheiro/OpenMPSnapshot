#include <omp.h>
#include <stdio.h>
#include <unistd.h>
int main(int argc, char* argv[])
{
int var = 0;
int i;
#pragma omp parallel sections 
{
for (i = 0; i < 10; i++) {
#pragma omp task shared(var) if(0)
{
var++;
}
}
}
printf("%d\n",var);
return 0;
}
