#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <map>
#include <cassert>
#include <omp.h>


int Fibonacci(int n) {
if (n < 2)
return n;
else
return Fibonacci(n-1) + Fibonacci(n-2);
}















int FibonacciOmp(int n) {

if (n < 2)
return n;

else if (n < 15) {
return FibonacciOmp(n-1) + FibonacciOmp(n-2);
}

else {
int val1, val2, res;

#pragma omp task shared(val1) firstprivate(n) priority(2)
val1 = FibonacciOmp(n-1);

val2 = FibonacciOmp(n-2);

#pragma omp taskwait
res = val1 + val2;

return res;

}

}



int FibonacciOmpDy(int n, std::map<int,int>& fiboMap) {

if (fiboMap.find(n) != fiboMap.end())
return fiboMap[n];

else {

if (n < 2)
fiboMap[n] = n;

else if (n < 20) {
fiboMap[n] = FibonacciOmpDy(n-1, fiboMap) + FibonacciOmpDy(n-2, fiboMap);
}

else {
int val1, val2;
#pragma omp task shared(val1) firstprivate(n)
val1 = FibonacciOmpDy(n-1, fiboMap);

val2 = FibonacciOmpDy(n-2, fiboMap);

#pragma omp taskwait
fiboMap[n] = val1 + val2;
}

return fiboMap[n];
}

}


void test(){
const long int TestSize = 40;
const long int NbLoops = 10;

std::cout << "Check Fibonacci" << std::endl;
std::cout << "TestSize = " << TestSize << std::endl;

int scalarFibonnaci = 0;
{
Timer timerSequential;

for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
scalarFibonnaci += Fibonacci(TestSize);
}
timerSequential.stop();

std::cout << ">> Sequential timer : " << timerSequential.getElapsed() << std::endl;
}
#pragma omp parallel
#pragma omp master
{

int ompFibonnaci = 0;
Timer timerParallel;

for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
ompFibonnaci += FibonacciOmp(TestSize);
}

timerParallel.stop();

std::cout << ">> There are " << omp_get_num_threads() << " threads" << std::endl;
std::cout << ">> Omp timer : " << timerParallel.getElapsed() << std::endl;

int ompFibonnaciDy = 0;
std::map<int,int> fiboMap;
timerParallel.reset();
for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
ompFibonnaciDy += FibonacciOmpDy(TestSize, fiboMap);
}
timerParallel.stop();

std::cout << ">> Omp timer dynamic : " << timerParallel.getElapsed() << std::endl;

CheckEqual(scalarFibonnaci,ompFibonnaci);
CheckEqual(scalarFibonnaci,ompFibonnaciDy);
}
}

int main(){
test();

return 0;
}