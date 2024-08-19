void f(int i, float f)
{
#pragma acc kernels num_gangs 
;
#pragma acc kernels num_workers 
;
#pragma acc kernels vector_length 
;
#pragma acc parallel num_gangs 
;
#pragma acc parallel num_workers 
;
#pragma acc parallel vector_length 
;
#pragma acc kernels num_gangs( 
;
#pragma acc kernels num_workers( 
;
#pragma acc kernels vector_length( 
;
#pragma acc parallel num_gangs( 
;
#pragma acc parallel num_workers( 
;
#pragma acc parallel vector_length( 
;
#pragma acc kernels num_gangs() 
;
#pragma acc kernels num_workers() 
;
#pragma acc kernels vector_length() 
;
#pragma acc parallel num_gangs() 
;
#pragma acc parallel num_workers() 
;
#pragma acc parallel vector_length() 
;
#pragma acc kernels num_gangs(1 
;
#pragma acc kernels num_workers(1 
;
#pragma acc kernels vector_length(1 
;
#pragma acc parallel num_gangs(1 
;
#pragma acc parallel num_workers(1 
;
#pragma acc parallel vector_length(1 
;
#pragma acc kernels num_gangs(i 
;
#pragma acc kernels num_workers(i 
;
#pragma acc kernels vector_length(i 
;
#pragma acc parallel num_gangs(i 
;
#pragma acc parallel num_workers(i 
;
#pragma acc parallel vector_length(i 
;
#pragma acc kernels num_gangs(1 i 
;
#pragma acc kernels num_workers(1 i 
;
#pragma acc kernels vector_length(1 i 
;
#pragma acc parallel num_gangs(1 i 
;
#pragma acc parallel num_workers(1 i 
;
#pragma acc parallel vector_length(1 i 
;
#pragma acc kernels num_gangs(1 i) 
;
#pragma acc kernels num_workers(1 i) 
;
#pragma acc kernels vector_length(1 i) 
;
#pragma acc parallel num_gangs(1 i) 
;
#pragma acc parallel num_workers(1 i) 
;
#pragma acc parallel vector_length(1 i) 
;
#pragma acc kernels num_gangs(1, i 
;
#pragma acc kernels num_workers(1, i 
;
#pragma acc kernels vector_length(1, i 
;
#pragma acc parallel num_gangs(1, i 
;
#pragma acc parallel num_workers(1, i 
;
#pragma acc parallel vector_length(1, i 
;
#pragma acc kernels num_gangs(1, i) 
;
#pragma acc kernels num_workers(1, i) 
;
#pragma acc kernels vector_length(1, i) 
;
#pragma acc parallel num_gangs(1, i) 
;
#pragma acc parallel num_workers(1, i) 
;
#pragma acc parallel vector_length(1, i) 
;
#pragma acc kernels num_gangs(num_gangs_k) 
;
#pragma acc kernels num_workers(num_workers_k) 
;
#pragma acc kernels vector_length(vector_length_k) 
;
#pragma acc parallel num_gangs(num_gangs_p) 
;
#pragma acc parallel num_workers(num_workers_p) 
;
#pragma acc parallel vector_length(vector_length_p) 
;
#pragma acc kernels num_gangs(f) 
;
#pragma acc kernels num_workers(f) 
;
#pragma acc kernels vector_length(f) 
;
#pragma acc parallel num_gangs(f) 
;
#pragma acc parallel num_workers(f) 
;
#pragma acc parallel vector_length(f) 
;
#pragma acc kernels num_gangs((float) 1) 
;
#pragma acc kernels num_workers((float) 1) 
;
#pragma acc kernels vector_length((float) 1) 
;
#pragma acc parallel num_gangs((float) 1) 
;
#pragma acc parallel num_workers((float) 1) 
;
#pragma acc parallel vector_length((float) 1) 
;
#pragma acc kernels num_gangs(0) 
;
#pragma acc kernels num_workers(0) 
;
#pragma acc kernels vector_length(0) 
;
#pragma acc parallel num_gangs(0) 
;
#pragma acc parallel num_workers(0) 
;
#pragma acc parallel vector_length(0) 
;
#pragma acc kernels num_gangs((int) -1.2) 
;
#pragma acc kernels num_workers((int) -1.2) 
;
#pragma acc kernels vector_length((int) -1.2) 
;
#pragma acc parallel num_gangs((int) -1.2) 
;
#pragma acc parallel num_workers((int) -1.2) 
;
#pragma acc parallel vector_length((int) -1.2) 
;
#pragma acc kernels num_gangs(1)  num_workers(1)  vector_length(1)  num_workers(1)  vector_length(1)  num_gangs(1) 
;
#pragma acc parallel							num_gangs(1)  num_workers(1)  vector_length(1)  num_workers(1)  vector_length(1)  num_gangs(1) 
;
#pragma acc kernels num_gangs(-1)  num_workers()  vector_length(abc_k)  num_workers(0.5)  vector_length(&f)  num_gangs( 
;
#pragma acc parallel							num_gangs(-1)  num_workers()  vector_length(abc_p)  num_workers(0.5)  vector_length(&f)  num_gangs( 
;
}
