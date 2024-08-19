#include "standardHeaders.h"
#include "splitUpWork.h"
#include "queue.h"
#include "readerThreadOMP.h"
#include "mapMPI.h"
#include <time.h>
#include <mpi.h>

omp_lock_t messLock;
omp_lock_t queLock;
omp_lock_t mapLock;
omp_lock_t redLock;
omp_lock_t writeLock;
omp_lock_t lk0;
omp_lock_t lk1;
omp_lock_t endL;

int readers; 
int mappers; 
int reducers; 
int writers;

int mapsRecieved = 0;
int reduceDone = 0; 

int main(int argc, char* argv[]){

int nodeRankNum; 
int clusterSize; 

MPI_Init(&argc, &argv); 
MPI_Comm_size(MPI_COMM_WORLD, &clusterSize);
MPI_Comm_rank(MPI_COMM_WORLD, &nodeRankNum);

omp_init_lock(&queLock);
omp_init_lock(&mapLock);
omp_init_lock(&redLock);  
omp_init_lock(&lk0);
omp_init_lock(&lk1);
omp_init_lock(&messLock);  
omp_init_lock(&endL);

int rCt;
int mCt;
int uCt;
int wCt;
int totalRead;
int toalMapped;


char cwd_old[PATH_MAX];
char cwd_new[PATH_MAX];

if (getcwd(cwd_old, sizeof(cwd_old)) != NULL) {
;
} else {
perror("getcwd() error");
}

strcpy(cwd_new, cwd_old);
strcat(cwd_new, "/RawText");
chdir(cwd_new);

printf("ClusterSize: %d\n", clusterSize);

MPI_Status requestMapTable;

int fileCount = 0;

struct que* fileQue = getNodeFileQue(clusterSize, nodeRankNum);

int threadNum = 3;
omp_set_num_threads(threadNum);

int mapDone[10];
int readDone[10];
int redDone[10];

int mapMpiDone = 0; 

for(int d = 0; d<10; d++)
{

mapDone[d] = 0;
readDone[d] = 0;
redDone[d] = 0;
}

#pragma omp barrier

double time = omp_get_wtime();

#pragma omp parallel
{
#pragma omp barrier
#pragma omp master
{

readers = 1; 
mappers = 1; 
reducers = 1; 
struct node* tempFile[readers];
struct que* q[readers];
struct mapChain* mapTables[mappers]; 
struct mapChain* masterTable; 
struct node* wordString[mappers];
int capacity = 163841;
struct que* readQue;
struct que* mapQue;
struct que* redQue;
struct que* writerQue;
char* mapTableString = (char*) malloc(clusterSize*2*capacity*25*sizeof(char));
char* tempTableString =  (char*) malloc(clusterSize*2*capacity*25*sizeof(char));
int* mapTableCount = (int*) malloc(capacity*sizeof(int));
char* tempPtr;

for(int mt = 0; mt< readers; mt++){
mapTables[mt] = initMapTable(capacity);
q[mt] = initQue();
}

masterTable = initMapTable(capacity);
readQue = initQue();
mapQue = initQue();
redQue = initQue();
writerQue = initQue();

int rCt = 0;
int mCt = 0;
int uCt = 0;
int wCt = 0;

int totalRead = 0;

double timeTest = omp_get_wtime();

for(int rr = 0; rr<readers; rr++)
{
#pragma omp task
{

int fqCt;
fqCt = rCt;

tempFile[fqCt] = dequ(fileQue);

while(!queEmpty(fileQue))
{

if(tempFile[fqCt] != NULL)
{
reader(q[fqCt], tempFile[fqCt]->str, lk0);
}

tempFile[fqCt] = dequ(fileQue);

}

readDone[rCt] = 1;
char readCount[2];
sprintf(readCount, "%d", rCt);
enqu(readQue, readCount);

}
rCt++;


}


for(int mm = 0; mm<mappers; mm++)
{

#pragma omp task
{
int mqCt;
mqCt = mCt;

int reDone = -1;
char rtDone[2];
sprintf(rtDone, "%d", mqCt);
while (mqCt != reDone)
{
if(!queEmpty(readQue))
{
omp_set_lock(&queLock);
struct node* readCheck = readQue->head;
while(readCheck != NULL)
{

if(strcmp(readCheck->str, rtDone) == 0)
{
reDone = mqCt;
}
readCheck = readCheck->next;
}
omp_unset_lock(&queLock);
}

}


omp_set_lock(&mapLock);
wordString[mqCt] = dequ(q[mqCt]);
omp_unset_lock(&mapLock);

while((!queEmpty(q[mqCt])))
{
if(wordString[mqCt] != NULL)
{
mapper(mapTables[mqCt], wordString[mqCt]->str, capacity);
}

wordString[mqCt] = dequ(q[mqCt]);

}


omp_set_lock(&mapLock);
mapDone[mCt] = 1;
enqu(mapQue, "DONE");
omp_unset_lock(&mapLock);

}
mCt++;

}



if(threadNum > 1)
{

for(int rd = 0; rd<reducers; rd++)
{
#pragma omp task
{
int rqCt;
rqCt = uCt;

int mapCt = 0;

while (mapCt < mappers)
{
if(!queEmpty(mapQue))
{

struct node* tempCheck = mapQue->head;
mapCt = 0;
omp_set_lock(&mapLock);
while(tempCheck != NULL)
{
mapCt++;
tempCheck = tempCheck->next;
}
omp_unset_lock(&mapLock);
}

}


for(int red = rqCt; red<capacity; red+=reducers)
{
for(int redmap = 0; redmap<mappers; redmap++)
{
struct mapNode* curChain = (struct mapNode*) mapTables[redmap][red].head;
if (curChain == NULL) {
; 
} else {
while (curChain != NULL){
combineWords(masterTable, curChain->mapStr,red,curChain->wordCount);  
curChain = curChain->next;
}
}



}


}
omp_set_lock(&redLock);
redDone[uCt] = uCt;
char redCount[2];
sprintf(redCount, "%d", uCt);
enqu(redQue, redCount);
omp_unset_lock(&redLock);

}
uCt++;

}
}

if(threadNum > 1)
{
int redCt = 0;

while (redCt < reducers)
{
if(!queEmpty(redQue))
{

struct node* redCheck = redQue->head;
redCt = 0;
omp_set_lock(&redLock);
while(redCheck != NULL)
{
redCt++;
redCheck = redCheck->next;
}
omp_unset_lock(&redLock);
}

}
}

printf("Node: %d, Pre Message Passing Time: %f\n", nodeRankNum, omp_get_wtime() - timeTest);

if(clusterSize > 1)
{
char* recString =  (char*) malloc(2*capacity*25*sizeof(char));
int* recCount =  (int*) malloc(capacity*sizeof(int));
tempPtr = mapTableString;
if(nodeRankNum != 0)
{
if(threadNum > 1)
{
convertMap(masterTable, tempTableString, mapTableCount, capacity, nodeRankNum, clusterSize);
} else {
convertMap(mapTables[0], tempTableString, mapTableCount, capacity, nodeRankNum, clusterSize);
}

}

if(nodeRankNum == 0) 
{
for(int nnum = 0; nnum<clusterSize; nnum++)
{
if(nnum != nodeRankNum)
{
MPI_Send(NULL, 0, MPI_INT, nnum, nodeRankNum*nnum + nnum, MPI_COMM_WORLD); 
MPI_Recv(recString, 2*capacity*25, MPI_CHAR, nnum, nodeRankNum*nnum + nnum, MPI_COMM_WORLD, &requestMapTable);
MPI_Recv(recCount, capacity, MPI_INT, nnum, nodeRankNum*nnum + nnum, MPI_COMM_WORLD, &requestMapTable);

if(threadNum >1)
{
mergeTable(masterTable, recString, recCount, capacity);
} else {
mergeTable(mapTables[0], recString, recCount, capacity);
}
}

}

} else {
MPI_Recv(NULL, 0, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &requestMapTable);
MPI_Send(tempTableString,  2*capacity*25, MPI_CHAR, requestMapTable.MPI_SOURCE, requestMapTable.MPI_TAG , MPI_COMM_WORLD);
MPI_Send(mapTableCount, capacity, MPI_INT, requestMapTable.MPI_SOURCE, requestMapTable.MPI_TAG, MPI_COMM_WORLD);
}



}
printf("Node: %d, End of Messages: %f\n", nodeRankNum, omp_get_wtime() - timeTest);
strcat(cwd_old, "/Output");
chdir(cwd_old);
if(nodeRankNum == 0)
{
if(threadNum > 1)
{
saveMapToFile(masterTable, capacity, nodeRankNum);
} else {
saveMapToFile(mapTables[0], capacity, nodeRankNum);
}

printf("Post Write Time: %f\n", omp_get_wtime() - timeTest);
}



}
#pragma omp barrier


}
omp_set_lock(&endL);
printf("Completed Node: %d, Time: %f\n", nodeRankNum, omp_get_wtime() - time);
omp_unset_lock(&endL);


MPI_Finalize();


return 0;

}
