#include "thread.h"
#include "field_manager.h"
#include <cstdlib>
#include <vector>
#include <iostream>
#include <omp.h>

Thread::Thread(int threadNumber, fieldType myInitialPart, fieldType myInitialBorders,
int numberOfIterations, 
FieldManager& manager, int numberOfThreads):
manager(manager),
threadNumber(threadNumber),
numberOfIterations(numberOfIterations),
numberOfThreads(numberOfThreads) {
waiting = true;
chunkHeight = myInitialPart.size();
chunkWigth = myInitialPart[0].size();
myPartWithBorders.push_back(myInitialBorders[0]);
for (int i = 0; i < chunkHeight; i++) {
myPartWithBorders.push_back(myInitialPart[i]);
}
myPartWithBorders.push_back(myInitialBorders[1]);
currentIteration = 0;
leftThread = NULL;
rightThread = NULL;
threadDescriptor = 1;
cancelled = false;
}

void Thread::getAdjacentThreads(Thread& leftThread, Thread& rightThread) {
this->leftThread = &leftThread;
this->rightThread = &rightThread;
}

void Thread::create() {
}

fieldType Thread::getComputedPart() {
fieldType computedPart;
for (int i = 0; i < chunkHeight; i++) {
computedPart.push_back(myPartWithBorders[i+1]);
}
return computedPart;
}
ll Thread::getCurrentIteration() {
return currentIteration;
}

void Thread::cancel(bool wait) {
waiting = wait;
cancelled = true;
}
void Thread::destroySemaphores() {

}


void* Thread::runInThread(void *thisThread) {
Thread* t = (Thread*) thisThread;
t->run();
return NULL;
}
void Thread::updateIterations(ll numberOfIterations) {
this->numberOfIterations = numberOfIterations + currentIteration;
}
void Thread::run() {
omp_set_num_threads(numberOfThreads);




while(currentIteration < numberOfIterations && !cancelled && !manager.wasStopped()) {
oneIteration();
currentIteration++;
}

}
void Thread::oneIteration() {
int sum;
fieldType myNewPart(myPartWithBorders);
#pragma omp parallel for
for (ll i = 1; i < chunkHeight + 1; i++) {
for (ll j = 0; j < chunkWigth; j++) {
sum = numberOfNeighbours(i, j);
if (myPartWithBorders[i][j]) {
myNewPart[i][j] = (sum == 2) || (sum == 3);
} else {
myNewPart[i][j] = (sum == 3);
}
}
}
myPartWithBorders = myNewPart;
ll size = myPartWithBorders.size();
std::vector<bool> tmp = myPartWithBorders[1];
myPartWithBorders[0] = myPartWithBorders[size - 2];
myPartWithBorders[size - 1] = tmp;
}

void Thread::exchangeBorders() {

}

int Thread::numberOfNeighbours(ll i, ll j) {
int sum = 0;
ll p,q;
for (int deltaI = -1; deltaI < 2; deltaI++) {
for (int deltaJ = -1; deltaJ < 2; deltaJ++) {
p = i+deltaI;
q = j+deltaJ;
if (p >= chunkHeight + 2) {
p = 0;
} else if (p < 0) {
p = chunkHeight + 1;
}
if (q >= chunkWigth) {
q = 0;
} else if (q < 0) {
q = chunkWigth - 1;
}
sum += myPartWithBorders[p][q];
}
}
sum -= myPartWithBorders[i][j];
return sum;
}
