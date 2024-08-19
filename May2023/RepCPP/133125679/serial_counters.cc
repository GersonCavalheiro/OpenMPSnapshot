
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
}

#pragma omp task depend(in: counter) shared(counter)
{
counter++;
}

#pragma omp taskwait
}
}

std::cout << "counter = " << counter << std::endl;
return 0;
}
