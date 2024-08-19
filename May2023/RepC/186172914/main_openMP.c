#include "standardHeaders.h"
#include "splitUpWork.h"
#include "queue.h"
#include "readerThreadOMP.h"
#include "mapOpenmp.h"
#include <time.h>

omp_lock_t messLock;
omp_lock_t queLock;
omp_lock_t mapLock;
omp_lock_t redLock;
omp_lock_t writeLock;
omp_lock_t lk0;
omp_lock_t lk1;

int readers; 
int mappers; 
int reducers; 
int writers;

int mapsRecieved = 0;
int reduceDone = 0; 

int main(int argc, char* argv[]){


int nodeRankNum = 0; 
int clusterSize = 1; 

omp_init_lock(&queLock);
omp_init_lock(&mapLock);
omp_init_lock(&redLock);  
omp_init_lock(&lk0);
omp_init_lock(&lk1);
omp_init_lock(&messLock);  

int rCt;
int mCt;
int uCt;
int wCt;
int totalRead;
int toalMapped;

/


}
#pragma omp barrier






}









return 0;

}

