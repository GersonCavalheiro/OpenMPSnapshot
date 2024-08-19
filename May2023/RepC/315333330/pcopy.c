void
f (char *cp)
{
#pragma acc parallel pcopy(cp[3:5])
;
}
