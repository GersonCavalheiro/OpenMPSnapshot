#include <stdio.h>
#include <openacc.h>
int
main (int argc, char *argv[])
{
int i;
acc_present_or_copyin (&i, sizeof i);
fprintf (stderr, "CheCKpOInT\n");
#pragma acc enter data create (i)
return 0;
}
