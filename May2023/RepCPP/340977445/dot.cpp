#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <omp.h>


double dot(const double* vec0, const double* vec1, const long int N){
double sum = 0;
for(long int idx = 0 ; idx < N ; ++idx){
sum += vec0[idx] * vec1[idx];
}
return sum;
}

double dotOmp(const double* vec0, const double* vec1, const long int N){
double sum = 0; 
#pragma omp parallel for reduction(+:sum)
for(long int idx = 0 ; idx < N ; ++idx){
sum += vec0[idx] * vec1[idx];
}
return sum;
}

void test(){
const long int TestSize = 500000;
const long int NbLoops = 1000;

std::cout << "Check dot" << std::endl;
std::cout << "TestSize = " << TestSize << std::endl;
std::cout << "There will be " << omp_get_max_threads() << " threads" << std::endl;

std::mt19937 gen(0);
std::uniform_real_distribution<double> dis(0, 1);

std::vector<double> vec0(TestSize);
std::vector<double> vec1(TestSize);

double currentSum = 0;
for(long int idxVal = 0 ; idxVal < TestSize ; ++idxVal){
vec0[idxVal] = dis(gen);
vec1[idxVal] = dis(gen);
currentSum += vec0[idxVal]*vec1[idxVal];
}
currentSum *= NbLoops;

{
double sequentialSum = 0;
Timer timerSequential;

for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
sequentialSum += dot(vec0.data(), vec1.data(), TestSize);
}
timerSequential.stop();

std::cout << ">> Sequential exec timer : " << timerSequential.getElapsed() << std::endl;

CheckEqual(currentSum,sequentialSum);
}
{
double parallelSum = 0;
Timer timerOmp;

for(long int idxLoop = 0 ; idxLoop < NbLoops ; ++idxLoop){
parallelSum += dotOmp(vec0.data(), vec1.data(), TestSize);
}
timerOmp.stop();

std::cout << ">> Parallel exec timer : " << timerOmp.getElapsed() << std::endl;

CheckEqual(currentSum,parallelSum);
}
}

int main(){
test();

return 0;
}