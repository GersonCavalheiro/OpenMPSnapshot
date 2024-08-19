#include <stdlib.h>
#include <openacc.h>
int
main (int argc, char *argv[])
{
{
if (!acc_on_device (acc_device_none))
abort ();
if (!acc_on_device (acc_device_host))
abort ();
if (acc_on_device (acc_device_not_host))
abort ();
if (acc_on_device (acc_device_nvidia))
abort ();
}
#pragma acc parallel if(0)
{
if (!acc_on_device (acc_device_none))
abort ();
if (!acc_on_device (acc_device_host))
abort ();
if (acc_on_device (acc_device_not_host))
abort ();
if (acc_on_device (acc_device_nvidia))
abort ();
}
#if !ACC_DEVICE_TYPE_host
#pragma acc parallel
{
if (acc_on_device (acc_device_none))
abort ();
if (acc_on_device (acc_device_host))
abort ();
if (!acc_on_device (acc_device_not_host))
abort ();
#if ACC_DEVICE_TYPE_nvidia
if (!acc_on_device (acc_device_nvidia))
abort ();
#else
if (acc_on_device (acc_device_nvidia))
abort ();
#endif
}
#endif
return 0;
}
