
#include <iostream>
#include <cstring>
#include <omp.h>

#if !defined(NTHREADS)
#define NTHREADS 2
#endif

float balance;

int token1;
int token2;
int token3;


int main() {

#pragma omp parallel num_threads(NTHREADS)
{
#pragma omp single
{
#pragma omp task depend(out:token1,token2,token3)
{
balance  = 1000; 
token1 = token2 = token3 = 1; 
}

#pragma omp task depend(in:token1)
{
float rate = 0.2; 
balance += (balance * rate); 
}

#pragma omp task depend(in:token2)
{
int amount = 200;
balance += amount;
}

#pragma omp task depend(in:token3)
{
int amount = 500; 
balance -= amount; 
}

#pragma omp taskwait
std::cout << "balance: " <<  balance << std::endl;
}
}
return 0;
}
