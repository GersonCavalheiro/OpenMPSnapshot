#include <stdlib.h>
#include <assert.h>
#include <openacc.h>
int
main ()
{
int *p = (int *)malloc (sizeof (int));
#pragma acc enter data copyin (p[0:1])
#pragma acc parallel present (p[0:1]) num_gangs (1)
{
p[0] = 1;
}
acc_copyout (p, sizeof (int));
assert (p[0] == 1);
acc_copyin (p, sizeof (int));
#pragma acc parallel present (p[0:1]) num_gangs (1)
{
p[0] = 2;
}
#pragma acc exit data copyout (p[0:1])
assert (p[0] == 2);
acc_copyin (p, sizeof (int));
#pragma acc parallel present (p[0:1]) num_gangs (1)
{
p[0] = 3;
}
acc_copyout (p, sizeof (int));
assert (p[0] == 3);
#pragma acc enter data copyin (p[0:1])
#pragma acc parallel present (p[0:1]) num_gangs (1)
{
p[0] = 3;
}
#pragma acc exit data copyout (p[0:1])
assert (p[0] == 3);
free (p);
return 0;
}
