void
f (char *cp)
{
#pragma acc parallel pcopyin(cp[4:6])
;
}
