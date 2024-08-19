#include <stdio.h>
int
main (int argc, char **argv)
{
int a[8] __attribute__((unused));
fprintf (stderr, "CheCKpOInT\n");
#pragma acc declare present (a)
}
