void f (void)
{
int i, j;
#pragma acc loop 
for(i = 1; i < 30; i++)
{
if (i == 7) goto out; 
#pragma acc loop
for(j = 5; j < 10; j++)
{
if (i == 6 && j == 7) goto out; 
}
}
out:
;
}
