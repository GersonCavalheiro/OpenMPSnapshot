#pragma acc routine gang
int
routine1 (int n)
{
int i;
#pragma acc loop
for (i = 0; i < n; i++)
;
return n + 1;
}
#pragma acc routine gang
int
routine2 (int n)
{
int i;
#pragma acc loop
for (i = 0; i < n; i++)
;
return n + 1;
}
int
main ()
{
int i;
#pragma acc parallel loop
for (i = 0; i < 8; i++)
;
#pragma acc parallel loop
for (i = 0; i < 8; i++)
;
return 0;
}
