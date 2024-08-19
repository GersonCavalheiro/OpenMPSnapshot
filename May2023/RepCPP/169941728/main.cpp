#include <iostream>
#include <omp.h>


#ifndef _OPENMP
fprintf(stderr, "OpenMP not supported");
#endif


void hello_world() {
#pragma omp parallel for
for(char i = 'a'; i <= 'z'; i++) {
std::cout << i << std::endl;
}
}


void reduction_sum() {
int sum = 0;
int array[10] = {1,2,3,4,5,6,7,8,9,10};

#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < 10; i++) {
sum += array[i];
}

std::cout << sum << std::endl;
}


void schedule() {

}


void library_routines() {
std::cout << "\n**************** Library Routines ****************\n";
std::cout << "CPU Number: " << omp_get_num_procs() << std::endl;

std::cout << "Parallel region 1: \n" << std::endl;
#pragma omp parallel
{
std::cout << "Num of threads is: " << omp_get_num_threads();
std::cout << "\nThis thread ID is: " << omp_get_thread_num() << std::endl;
};

std::cout << "Parallel region 2: \n" << std::endl;
omp_set_num_threads(4);
#pragma omp parallel
{
std::cout << "Num of threads is: " << omp_get_num_threads();
std::cout << "\nThis thread ID is: " << omp_get_thread_num() << std::endl;
};
}

int main() {


hello_world();


reduction_sum();


library_routines();

return 0;
}