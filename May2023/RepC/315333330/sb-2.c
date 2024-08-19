void foo(int i)
{
switch (i) 
{
#pragma acc parallel 
{ case 0:; }
}
switch (i) 
{
#pragma acc kernels 
{ case 0:; }
}
switch (i) 
{
#pragma acc data 
{ case 0:; }
}
}
