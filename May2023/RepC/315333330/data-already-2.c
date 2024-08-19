#include <stdio.h>
int
main (int argc, char *argv[])
{
int i;
#pragma acc data present_or_copy (i)
{
fprintf (stderr, "CheCKpOInT\n");
#pragma acc data copyout (i)
++i;
}
return 0;
}
