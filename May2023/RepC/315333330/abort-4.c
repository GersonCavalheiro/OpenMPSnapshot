#include <stdlib.h>
int
main (int argc, char **argv)
{
#pragma acc kernels
{
if (argc != 1)
abort ();
}
return 0;
}
