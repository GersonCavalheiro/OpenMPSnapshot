int A[1024];
int B[1024];
int A_indexes[4];
int A_sizes[4];
int main() {
A_indexes[0] = 0;
A_sizes[0] = 25;
A_indexes[1] = 25;
A_sizes[1] = 200;
A_indexes[2] = 225;
A_sizes[2] = 500;
A_indexes[3] = 725;
A_sizes[3] = 524;
#pragma omp task in(B) inout( { A[ A_indexes[i] ; A_sizes[i] ] , i = 0 ; 4 } )
{
}
#pragma omp taskwait
return 0;
}
