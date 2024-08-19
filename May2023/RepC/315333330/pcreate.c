void
f (char *cp)
{
#pragma acc parallel pcreate(cp[6:8])
;
}
