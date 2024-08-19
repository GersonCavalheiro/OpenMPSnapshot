#include <stdio.h>
#include <openacc.h>
int
main (int argc, char *argv[])
{
int i;
#pragma acc enter data create (i)
fprintf (stderr, "CheCKpOInT\n");
acc_copyin (&i, sizeof i);
return 0;
}
