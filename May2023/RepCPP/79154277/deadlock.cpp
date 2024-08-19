

#include <omp.h>

#include <vector>
#include <iostream>

void foo()
{
#pragma omp parallel
#pragma omp critical
{
auto id = omp_get_thread_num();
std::cout << "o,hai there, i'm " << id << '\n';
}

}

void deadlockA()
{
int sum{0}; 

#pragma omp parallel for
for (int ii=0; ii<100; ++ii)
{
#pragma omp critical
{
sum += ii;
foo();
}
}
}


void deadlockB()
{
#pragma omp parallel
{
#pragma omp critical(A)
{
#pragma omp critical(B)
{
std::cout << "whassup\n";
}
}
#pragma omp critical(B)
{
#pragma omp critical(A)
{
std::cout << "ahoy\n";
}
}
}

}



int main()
{
deadlockB();


return 0;
}
