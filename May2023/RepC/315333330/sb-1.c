void foo()
{
int l;
bad1:
#pragma acc parallel
goto bad1; 
#pragma acc kernels
goto bad1; 
#pragma acc data
goto bad1; 
#pragma acc loop 
for (l = 0; l < 2; ++l)
goto bad1; 
goto bad2_parallel; 
#pragma acc parallel
{
bad2_parallel: ;
}
goto bad2_kernels; 
#pragma acc kernels
{
bad2_kernels: ;
}
goto bad2_data; 
#pragma acc data
{
bad2_data: ;
}
goto bad2_loop; 
#pragma acc loop 
for (l = 0; l < 2; ++l)
{
bad2_loop: ;
}
#pragma acc parallel
{
int i;
goto ok1_parallel;
for (i = 0; i < 10; ++i)
{ ok1_parallel: break; }
}
#pragma acc kernels
{
int i;
goto ok1_kernels;
for (i = 0; i < 10; ++i)
{ ok1_kernels: break; }
}
#pragma acc data
{
int i;
goto ok1_data;
for (i = 0; i < 10; ++i)
{ ok1_data: break; }
}
#pragma acc loop 
for (l = 0; l < 2; ++l)
{
int i;
goto ok1_loop;
for (i = 0; i < 10; ++i)
{ ok1_loop: break; }
}
}
