int
main ()
{
int a, i;
#pragma acc parallel loop vector copy(a[0:100]) reduction(+:a) 
for (i = 0; i < 100; i++)
a++;
return a;
}
