void f(int i)
{
#pragma acc kernels num_gangs(i) num_workers(i) vector_length(i)
;
#pragma acc parallel num_gangs(i) num_workers(i) vector_length(i)
;
}
