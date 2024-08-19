void acc_parallel()
{
int i, j, k;
#pragma acc parallel num_gangs(i) 
;
#pragma acc parallel num_workers(j) 
;
#pragma acc parallel vector_length(k) 
;
}
void acc_kernels()
{
int i, j, k;
#pragma acc kernels num_gangs(i) 
;
#pragma acc kernels num_workers(j) 
;
#pragma acc kernels vector_length(k) 
;
}
