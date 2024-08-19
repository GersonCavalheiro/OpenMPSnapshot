void
foo (void)
{
int i;
#pragma acc host_data use_device(i) 
{
}
}
