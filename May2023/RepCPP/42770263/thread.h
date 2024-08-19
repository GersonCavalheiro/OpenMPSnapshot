#pragma once
#include "../common_lib/headers/typedefs.h"
#include <pthread.h>
#include <semaphore.h>
#include <vector>

class FieldManager;

class Thread {
pthread_t threadDescriptor;
int threadNumber;
ll chunkWigth;
ll chunkHeight;
Thread* leftThread; 
Thread* rightThread; 
sem_t* leftSemaphore; 
sem_t* rightSemaphore; 
sem_t* leftControlSemaphore; 
sem_t* rightControlSemaphore; 
pthread_cond_t* stopped; 
pthread_mutex_t* stopMutex; 
ll numberOfIterations;
ll currentIteration;
fieldType myPartWithBorders; 
FieldManager& manager;
bool cancelled;
bool waiting;
public:
Thread(int threadNumber, fieldType myInitialPart, fieldType initialBorders,
int numberOfIterations, pthread_cond_t* stopped, pthread_mutex_t* stopMutex,
FieldManager& manager);

void getAdjacentThreads(Thread& leftThread, Thread& rightThread);

void create();

void cancel(bool wait);

fieldType getComputedPart();

ll getCurrentIteration();

void updateIterations(ll numberOfIterations);

private:
static void* runInThread(void* thisThread);
void run();
void oneIteration();

void exchangeBorders();

int numberOfNeighbours(ll i, ll j);

void destroySemaphores();

};
