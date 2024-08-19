int
main (void)
{
#pragma acc parallel
{
int j = 0;
#pragma acc loop private (j)
for (int i = 0; i < 10; i++)
j++;
}
return 0;
}
