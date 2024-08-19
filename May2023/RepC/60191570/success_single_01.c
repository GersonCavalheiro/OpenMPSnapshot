#include <stdlib.h>
int main(int argc, char* argv[])
{
char people = 0;
#pragma omp single
{
people = 1;
}
if (people != 1)
abort();
return 0;
}
