
#include <iostream>
#include <omp.h>

int main() {
int counter = 0;

#pragma omp parallel num_threads(4) shared(counter)
{
#pragma omp master
{
#pragma omp task depend(out: counter) shared(counter)
{
counter++;
std::cout << "Counter 1 Thread: "
<< omp_get_thread_num() << std::endl;
}

#pragma omp task depend(in: counter) shared(counter)
{
counter++;
std::cout << "Counter 2 Thread: "
<< omp_get_thread_num() << std::endl;
}

#pragma omp taskwait
}
}

std::cout << "counter = " << counter << std::endl;
return 0;
}
