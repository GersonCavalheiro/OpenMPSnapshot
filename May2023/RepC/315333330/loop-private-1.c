void
f (int i, int j)
{
#pragma acc kernels
#pragma acc loop collapse(2)
for (i = 0; i < 20; ++i)
for (j = 0; j < 25; ++j)
;
}
