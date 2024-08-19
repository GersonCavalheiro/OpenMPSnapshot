#include <stdio.h>
#include <openacc.h>
int
main (int argc, char *argv[])
{
int i;
#pragma acc data present_or_copy (i)
{
fprintf (stderr, "CheCKpOInT\n");
acc_copyin (&i, sizeof i);
}
return 0;
}
