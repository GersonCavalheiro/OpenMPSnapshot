void
f (char *cp)
{
#pragma acc parallel pcopyout(cp[5:7])
;
}
