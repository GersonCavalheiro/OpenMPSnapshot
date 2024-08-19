#include <omp.h>
#include <stdio.h>
#include <string>
void print_funcs(std::string name){
printf("%-25s | num_threads = %d  | thread_num = %d | num_procs = %d\n", name.c_str(), omp_get_num_threads(), omp_get_thread_num(), omp_get_num_procs());
printf("------\n");
}
int main(){
print_funcs("Before parallel region");
#pragma omp parallel
{
print_funcs("In parallel region");
}
print_funcs("After parallel region");
return 0;
}
