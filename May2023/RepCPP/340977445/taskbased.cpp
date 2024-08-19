#include <iostream>
#include <omp.h>
#include <cassert>
#include <vector>
#include <random>
#include <unistd.h>
#include "timer.hpp"

void kernel(const long int& r0, const long int& r1, long int& w0){
usleep(100000);
w0 += r0 * r1;
}

void DumbExecution(std::vector<long int>& inElements, const long int nbTasks){
if(inElements.size() == 0){
return;
}

std::mt19937 gen(0); 
std::uniform_int_distribution<> dis(0, int(inElements.size()-1));

for(long int idxTask = 0 ; idxTask < nbTasks ; ++idxTask){
const int idxRead0 = dis(gen);
const int idxRead1 = dis(gen);
const int idxWrite0 = dis(gen);

kernel(inElements[idxRead0], inElements[idxRead1], inElements[idxWrite0]);
}
}

void DumbExecutionOmp(std::vector<long int>& inElements, const long int nbTasks){
if(inElements.size() == 0){
return;
}

long int* inElementsPtr = inElements.data();

#pragma omp parallel
#pragma omp master
{    
#pragma omp taskgroup
{
std::mt19937 gen(0); 
std::uniform_int_distribution<> dis(0, int(inElements.size()-1));

for(long int idxTask = 0 ; idxTask < nbTasks ; ++idxTask){
const int idxRead0 = dis(gen);
const int idxRead1 = dis(gen);
const int idxWrite0 = dis(gen);
#pragma omp task firstprivate(idxRead0, idxRead1, idxWrite0) depend(in:idxRead0, idxRead1) depend(out:idxWrite0) 
{
kernel(inElementsPtr[idxRead0], inElementsPtr[idxRead1], inElementsPtr[idxWrite0]);    
}
}
}
}  
}

void test(){
const long int TestSize = 200;
const long int NbTasks = 100;

std::cout << "Check taskbased" << std::endl;
std::cout << "TestSize = " << TestSize << std::endl;
std::cout << "NbTasks = " << NbTasks << std::endl;

std::vector<long int> elements(TestSize);
std::fill_n(elements.begin(), elements.size(), 1);

std::vector<long int> elementsOmp = elements;

{
Timer timerSequential;
DumbExecution(elements, NbTasks);
timerSequential.stop();

std::cout << ">> Sequential timer : " << timerSequential.getElapsed() << std::endl;
}
{
Timer timerParallel;
DumbExecutionOmp(elementsOmp, NbTasks);
timerParallel.stop();

std::cout << ">> Omp timer : " << timerParallel.getElapsed() << std::endl;
}

if(elements == elementsOmp){
std::cout << "Execution succeed!" << std::endl;
}
else{
std::cout << "Execution failled! Vectors are different sorry" << std::endl;
}
}

int main(){
test();

return 0;
}