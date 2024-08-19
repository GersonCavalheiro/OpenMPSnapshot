#include <stdio.h>
int
main (int argc, char *argv[])
{
int i;
#pragma acc data create (i)
{
fprintf (stderr, "CheCKpOInT\n");
#pragma acc parallel copyin (i)
++i;
}
return 0;
}
