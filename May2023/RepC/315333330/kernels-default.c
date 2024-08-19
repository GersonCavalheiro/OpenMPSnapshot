void
foo (void)
{
unsigned int i = 0;
#pragma acc kernels
{
i++;
}
}
