




#pragma once
#include <iostream>
#include <cmath>
#include <mpi.h>
#include <math.h>
#include <iomanip>
#include <queue>
#include "commonLib.h"
#include "linearpart.h"
#include "createpart.h"
#include "partition.h"
#include "tiffIO.h"

long LAST_ID = -1;

struct point {
long x;
long y;
float elev;
float area;
float length;
};

struct streamlink {
int32_t Id;
int32_t u1;
int32_t u2;
long d;
long magnitude;
long shapeId;
double elevU;
double elevD;
double length;
short order;
queue <point> coord; 
long numCoords;
bool terminated;

};

struct llnode {
streamlink *data;
llnode *next;
};
struct LinkedLL {
llnode *head;
int numLinks;
};

LinkedLL linkSet;
void makeLinkSet() {
linkSet.head = NULL;
linkSet.numLinks = 0;
return;
}
void setLinkInfo(long **LinkIdU1U2DMagShapeid,
double **LinkElevUElevDLength,
double **PointXY,
float **PointElevArea,
tdpartition *elev,
tiffIO *elevIO);
void getNumLinksAndPoints(long &myNumLinks, long &myNumPoints);
void SendAndReciveLinks(int nx,
int ny,
tdpartition *idGrid,
tdpartition *contribs,
tdpartition *flowDir,
tdpartition *src);
long getMagnitude(int32_t Id);
void appendPoint(int32_t Id, point *addPoint);
void setDownLinkId(int32_t Id, long dId);
streamlink *createLink(int32_t u1, int32_t u2, long d, point *coord); 
void linkSetInsert(streamlink *linkToAdd);
long GetOrderOf(long ID);
long findLinkThatEndsAt(long x, long y, tdpartition *);
bool recvLink(int src);
bool sendLink(int32_t Id, int dest);
void terminateLink(int32_t Id);
streamlink *FindLink(int32_t Id);
streamlink *getFirstLink();
streamlink *takeOut(int32_t Id);
llnode *LSInsert(llnode *head, llnode *addNode);

llnode *LSInsert(llnode *head, llnode *addNode) {
addNode->next = head;
head = addNode;
return head;
}

streamlink *takeOut(int32_t Id) {
streamlink *linkToBeRemoved;
llnode *nodeToBeRemoved;
if (linkSet.head == NULL) {
return NULL;
}
if (linkSet.head->data->Id == Id) {
linkToBeRemoved = linkSet.head->data;
nodeToBeRemoved = linkSet.head;
linkSet.head = linkSet.head->next;
delete nodeToBeRemoved;
linkSet.numLinks--;
return linkToBeRemoved;
} else {
llnode *previous, *current;
current = linkSet.head;
previous = NULL;
while (current->data->Id != Id && current->next != NULL) {
previous = current;
current = current->next;
}
if (current->data->Id == Id) {
previous->next = current->next;
nodeToBeRemoved = current;
linkToBeRemoved = current->data;
delete nodeToBeRemoved;
linkSet.numLinks--;
return linkToBeRemoved;
}
}
return NULL;
}
streamlink *FindLink(int32_t Id) {
streamlink *linkToBeRemoved;
llnode *nodeToBeRemoved;
if (linkSet.head == NULL) {
return NULL;
}
if (linkSet.head->data->Id == Id) {
return linkSet.head->data;
} else {
llnode *previous, *current;
current = linkSet.head;
previous = NULL;
while (current->data->Id != Id && current->next != NULL) {
previous = current;
current = current->next;
}
if (current->data->Id == Id) {
return current->data;
}
}
return NULL;
}
void terminateLink(int32_t Id) {
streamlink *linkToKill;
linkToKill = FindLink(Id);
linkToKill = linkToKill;
if (linkToKill == NULL) {
MPI_Abort(MCW, 4);
} else {
linkToKill->terminated = true;
}
return;
}




bool sendLink(int32_t Id, int dest) {
int rank, size;
MPI_Comm_rank(MCW, &rank);
MPI_Comm_size(MCW, &size);

streamlink *toSend = takeOut(Id);
if (toSend == NULL) {
return false;
}
if (toSend->terminated == true) {
linkSetInsert(toSend);
return false;
}

MPI_Request req;
MPI_Send(&(toSend->Id), 1, MPI_LONG, dest, 1, MCW);
MPI_Send(&(toSend->u1), 1, MPI_LONG, dest, 2, MCW);
MPI_Send(&(toSend->u2), 1, MPI_LONG, dest, 3, MCW);
MPI_Send(&(toSend->d), 1, MPI_LONG, dest, 4, MCW);
MPI_Send(&(toSend->elevU), 1, MPI_DOUBLE, dest, 5, MCW);
MPI_Send(&(toSend->elevD), 1, MPI_DOUBLE, dest, 6, MCW);
MPI_Send(&(toSend->length), 1, MPI_DOUBLE, dest, 7, MCW);
MPI_Send(&(toSend->order), 1, MPI_SHORT, dest, 8, MCW);
MPI_Send(&(toSend->numCoords), 1, MPI_LONG, dest, 9, MCW);
MPI_Send(&(toSend->magnitude), 1, MPI_LONG, dest, 11, MCW);
MPI_Send(&(toSend->shapeId), 1, MPI_LONG, dest, 12, MCW);

MPI_Datatype PointType, oldtypes[2];
int blockcounts[2];

MPI_Aint offsets[2], lb, extent;
MPI_Status stat;
offsets[0] = 0;
oldtypes[0] = MPI_LONG;
blockcounts[0] = 2;
MPI_Type_get_extent(MPI_LONG, &lb, &extent);
offsets[1] = 2 * extent;
oldtypes[1] = MPI_FLOAT;
blockcounts[1] = 3;
MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &PointType);
MPI_Type_commit(&PointType);

MPI_Status status;

char *ptr;
int place;
point *buf;
point *sendArr;
sendArr = new point[toSend->numCoords];
for (int i = 0; i < toSend->numCoords; i++) {
sendArr[i].x = toSend->coord.front().x;
sendArr[i].y = toSend->coord.front().y;
sendArr[i].elev = toSend->coord.front().elev;
sendArr[i].area = toSend->coord.front().area;
sendArr[i].length = toSend->coord.front().length;
toSend->coord.pop();
}

int bsize = toSend->numCoords * sizeof(point) * 2
+ MPI_BSEND_OVERHEAD;  
buf = new point[bsize];

MPI_Buffer_attach(buf, bsize);
MPI_Bsend(sendArr, toSend->numCoords, PointType, dest, 10, MCW);
MPI_Buffer_detach(&ptr, &place);
delete sendArr;
delete toSend;

return true;
}
bool recvLink(int src) {
streamlink *toRecv = new streamlink;
MPI_Status stat;
int rank, size;
MPI_Comm_rank(MCW, &rank);
MPI_Comm_size(MCW, &size);

MPI_Recv(&(toRecv->Id), 1, MPI_LONG, src, 1, MCW, &stat);
MPI_Recv(&(toRecv->u1), 1, MPI_LONG, src, 2, MCW, &stat);
MPI_Recv(&(toRecv->u2), 1, MPI_LONG, src, 3, MCW, &stat);
MPI_Recv(&(toRecv->d), 1, MPI_LONG, src, 4, MCW, &stat);
MPI_Recv(&(toRecv->elevU), 1, MPI_DOUBLE, src, 5, MCW, &stat);
MPI_Recv(&(toRecv->elevD), 1, MPI_DOUBLE, src, 6, MCW, &stat);
MPI_Recv(&(toRecv->length), 1, MPI_DOUBLE, src, 7, MCW, &stat);
MPI_Recv(&(toRecv->order), 1, MPI_SHORT, src, 8, MCW, &stat);
MPI_Recv(&(toRecv->numCoords), 1, MPI_LONG, src, 9, MCW, &stat);
MPI_Recv(&(toRecv->magnitude), 1, MPI_LONG, src, 11, MCW, &stat);
MPI_Recv(&(toRecv->shapeId), 1, MPI_LONG, src, 12, MCW, &stat);

MPI_Datatype PointType, oldtypes[2];
int blockcounts[2];

MPI_Aint offsets[2], lb, extent;
MPI_Status stat1;
offsets[0] = 0;
oldtypes[0] = MPI_LONG;
blockcounts[0] = 2;
MPI_Type_get_extent(MPI_LONG, &lb, &extent);
offsets[1] = 2 * extent;
oldtypes[1] = MPI_FLOAT;
blockcounts[1] = 3;
MPI_Type_create_struct(2, blockcounts, offsets, oldtypes, &PointType);
MPI_Type_commit(&PointType);
int flag;

point *recvArr;
recvArr = new point[toRecv->numCoords];
MPI_Recv(recvArr, toRecv->numCoords, PointType, src, 10, MCW, &stat);
toRecv->terminated = false;

point temp;
for (int i = 0; i < toRecv->numCoords > 0; i++) {
temp.x = recvArr[i].x;
temp.y = recvArr[i].y;
temp.elev = recvArr[i].elev;
temp.area = recvArr[i].area;
temp.length = recvArr[i].length;
toRecv->coord.push(temp);
}

linkSetInsert(toRecv);
delete recvArr;

return true;
}
long findLinkThatEndsAt(long x, long y, tdpartition *elev) {
long ID = -1;
llnode *current = linkSet.head;
if (linkSet.numLinks == 0) {
return ID;
} else {
int gx, gy;
elev->localToGlobal((int) x, (int) y, gx, gy);
while (current != NULL) {
if (current->data->coord.back().x == gx && current->data->coord.back().y == gy
&& !current->data->terminated) {
ID = current->data->Id;
break;
}
current = current->next;
}
}
return ID;
}
long GetOrderOf(long ID) {
llnode *current = linkSet.head;
if (linkSet.numLinks == 0) {
return -1;
} else {
while (current != NULL) {
if (ID == current->data->Id) {
return current->data->order;
}
current = current->next;
}
}
return -1;
}

void linkSetInsert(streamlink *linkToAdd) {
llnode *newNode = new llnode;
newNode->data = linkToAdd;
newNode->next = NULL;

linkSet.numLinks++;
linkSet.head = LSInsert(linkSet.head, newNode);
return;
}
streamlink *createLink(int32_t u1, int32_t u2, long d, point *coord) 
{
int size;
MPI_Comm_size(MCW, &size);
int rank;
MPI_Comm_rank(MCW, &rank);

streamlink *newLink = new streamlink;

if (LAST_ID == -1) {
newLink->Id = rank;
} else {
newLink->Id = LAST_ID + size;
}
LAST_ID = newLink->Id;

newLink->u1 = u1;    
newLink->u2 = u2;    
newLink->d = d;    

newLink->coord.push(*coord);


newLink->numCoords = 1; 
newLink->elevU = coord->elev;  
newLink->elevD = coord->elev;  
newLink->order = 1;
newLink->terminated = false;

newLink->magnitude = 1;
int i = 0;
newLink->length = 0;
newLink->shapeId = -1;
linkSetInsert(newLink);
return newLink;
}

void setDownLinkId(int32_t Id, long dId) {
streamlink *myLink = FindLink(Id);
myLink->d = dId;
return;
}
void appendPoint(int32_t Id, point *addPoint) {
streamlink *myLink = FindLink(Id);
if (myLink == NULL) {
MPI_Abort(MCW, 8);
}
myLink->numCoords = myLink->numCoords + 1;

long i;
myLink->coord.push(*addPoint);
myLink->elevD = addPoint->elev;
return;
}
void SendAndReciveLinks(int nx,
int ny,
tdpartition *idGrid,
tdpartition *contribs,
tdpartition *flowDir,
tdpartition *src) {
int linksSent = 0;
int linksRecv = 0;
int totalSent = 0;
int totalRecv = 0;
int toSend = 0;
int toRecv = 0;

short tempShort;
int32_t tempLong;

int rank, size, sent;
MPI_Comm_rank(MCW, &rank);
MPI_Comm_size(MCW, &size);
MPI_Status stat;
flowDir->share();
if (size == 1) {
return;
}
if (rank % 2 == 0) {
int i;
int inext, jnext;
if (rank != 0) {
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, -1) && contribs->getData(i, -1, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) 0, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) 0, tempShort) == 2
&& src->getData(i - 1, 0, tempShort) == 1 && idGrid->getData(i - 1, 0, tempLong) >= 0) {
toSend++;
} else if (flowDir->getData((long) i, (long) 0, tempShort) == 3
&& src->getData(i, 0, tempShort) == 1 && idGrid->getData(i, 0, tempLong) >= 0) {
toSend++;
} else if (i < nx && flowDir->getData((long) i + 1, (long) 0, tempShort) == 4
&& src->getData(i + 1, 0, tempShort) == 1 && idGrid->getData(i + 1, 0, tempLong) >= 0) {
toSend++;
}
}
}
MPI_Send(&toSend, 1, MPI_INT, rank - 1, rank, MCW);
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, -1) && contribs->getData(i, -1, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) 0, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) 0, tempShort) == 2
&& src->getData(i - 1, 0, tempShort) == 1 && idGrid->getData(i - 1, 0, tempLong) >= 0) {
if (sendLink(idGrid->getData(i - 1, 0, tempLong), rank - 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
} else if (flowDir->getData((long) i, (long) 0, tempShort) == 3
&& src->getData(i, 0, tempShort) == 1 && idGrid->getData(i, 0, tempLong) >= 0) {
if (sendLink(idGrid->getData(i, 0, tempLong), rank - 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
} else if (i < nx && flowDir->getData((long) i + 1, (long) 0, tempShort) == 4
&& src->getData(i + 1, 0, tempShort) == 1 && idGrid->getData(i + 1, 0, tempLong) >= 0) {
if (sendLink(idGrid->getData(i + 1, 0, tempLong), rank - 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
}
}
}
}
toSend = 0;
if (rank != size - 1) {
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, ny) && contribs->getData(i, ny, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) ny - 1, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) ny - 1, tempShort) == 8
&& src->getData(i - 1, ny - 1, tempShort) == 1
&& idGrid->getData(i - 1, ny - 1, tempLong) >= 0) {
toSend++;
} else if (flowDir->getData((long) i, (long) ny - 1, tempShort) == 7
&& src->getData(i, ny - 1, tempShort) == 1 && idGrid->getData(i, ny - 1, tempLong) >= 0) {
toSend++;
} else if (i < nx && flowDir->getData((long) i + 1, (long) ny - 1, tempShort) == 6
&& src->getData(i + 1, ny - 1, tempShort) == 1
&& idGrid->getData(i + 1, ny - 1, tempLong) >= 0) {
toSend++;
}
}
}
MPI_Send(&toSend, 1, MPI_INT, rank + 1, rank, MCW);
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, ny) && contribs->getData(i, ny, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) ny - 1, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) ny - 1, tempShort) == 8
&& src->getData(i - 1, ny - 1, tempShort) == 1
&& idGrid->getData(i - 1, ny - 1, tempLong) >= 0) {
if (sendLink(idGrid->getData(i - 1, ny - 1, tempLong),
rank + 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
} else if (flowDir->getData((long) i, (long) ny - 1, tempShort) == 7
&& src->getData(i, ny - 1, tempShort) == 1 && idGrid->getData(i, ny - 1, tempLong) >= 0) {
if (sendLink(idGrid->getData(i, ny - 1, tempLong), rank + 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
} else if (i < nx && flowDir->getData((long) i + 1, (long) ny - 1, tempShort) == 6
&& src->getData(i + 1, ny - 1, tempShort) == 1
&& idGrid->getData(i + 1, ny - 1, tempLong) >= 0) {
if (sendLink(idGrid->getData(i + 1, ny - 1, tempLong),
rank + 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
}
}
}
}
} else {
int i = 0;
if (rank != 0) {
MPI_Recv(&toRecv, 1, MPI_INT, rank - 1, rank - 1, MCW, &stat);
int i = 0;
for (i = 0; i < toRecv; i++) {
if (recvLink(rank - 1)) {
linksRecv++;
} else {
MPI_Abort(MCW, 2);
}
}
}
i = 0;
if (rank != size - 1) {
MPI_Recv(&toRecv, 1, MPI_INT, rank + 1, rank + 1, MCW, &stat);
int i = 0;
for (i = 0; i < toRecv; i++) {
if (recvLink(rank + 1)) {
linksRecv++;
} else {
MPI_Abort(MCW, 2);
}
}
}
}
MPI_Barrier(MCW);

if (rank % 2 == 1) {
int i;
int inext, jnext;
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, -1) && contribs->getData(i, -1, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) 0, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) 0, tempShort) == 2
&& src->getData(i - 1, 0, tempShort) == 1 && idGrid->getData(i - 1, 0, tempLong) >= 0) {
toSend++;
}
if (flowDir->getData((long) i, (long) 0, tempShort) == 3 && src->getData(i, 0, tempShort) == 1
&& idGrid->getData(i, 0, tempLong) >= 0) {
toSend++;
}
if (i < nx && flowDir->getData((long) i + 1, (long) 0, tempShort) == 4
&& src->getData(i + 1, 0, tempShort) == 1 && idGrid->getData(i + 1, 0, tempLong) >= 0) {
toSend++;
}
}
}
MPI_Send(&toSend, 1, MPI_INT, rank - 1, rank, MCW);
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, -1) && contribs->getData(i, -1, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) 0, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) 0, tempShort) == 2
&& src->getData(i - 1, 0, tempShort) == 1 && idGrid->getData(i - 1, 0, tempLong) >= 0) {
if (sendLink(idGrid->getData(i - 1, 0, tempLong), rank - 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
}
if (flowDir->getData((long) i, (long) 0, tempShort) == 3 && src->getData(i, 0, tempShort) == 1
&& idGrid->getData(i, 0, tempLong) >= 0) {
if (sendLink(idGrid->getData(i, 0, tempLong), rank - 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
}
if (i < nx && flowDir->getData((long) i + 1, (long) 0, tempShort) == 4
&& src->getData(i + 1, 0, tempShort) == 1 && idGrid->getData(i + 1, 0, tempLong) >= 0) {
if (sendLink(idGrid->getData(i + 1, 0, tempLong), rank - 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
}
}
}
toSend = 0;
if (rank != size - 1) {
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, ny) && contribs->getData(i, ny, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) ny - 1, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) ny - 1, tempShort) == 8
&& src->getData(i - 1, ny - 1, tempShort) == 1
&& idGrid->getData(i - 1, ny - 1, tempLong) >= 0) {
toSend++;
} else if (flowDir->getData((long) i, (long) ny - 1, tempShort) == 7
&& src->getData(i, ny - 1, tempShort) == 1 && idGrid->getData(i, ny - 1, tempLong) >= 0) {
toSend++;
} else if (i < nx && flowDir->getData((long) i + 1, (long) ny - 1, tempShort) == 6
&& src->getData(i + 1, ny - 1, tempShort) == 1
&& idGrid->getData(i + 1, ny - 1, tempLong) >= 0) {
toSend++;
}
}
}
MPI_Send(&toSend, 1, MPI_INT, rank + 1, rank, MCW);
for (i = 0; i < nx; i++) {
if (!contribs->isNodata(i, ny) && contribs->getData(i, ny, tempShort)
< 0) {
int p = flowDir->getData((long) i - 1, (long) ny - 1, tempShort);
if (i > 0 && flowDir->getData((long) i - 1, (long) ny - 1, tempShort) == 8
&& src->getData(i - 1, ny - 1, tempShort) == 1
&& idGrid->getData(i - 1, ny - 1, tempLong) >= 0) {
if (sendLink(idGrid->getData(i - 1, ny - 1, tempLong),
rank + 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
} else if (flowDir->getData((long) i, (long) ny - 1, tempShort) == 7
&& src->getData(i, ny - 1, tempShort) == 1 && idGrid->getData(i, ny - 1, tempLong) >= 0) {
if (sendLink(idGrid->getData(i, ny - 1, tempLong), rank + 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
} else if (i < nx && flowDir->getData((long) i + 1, (long) ny - 1, tempShort) == 6
&& src->getData(i + 1, ny - 1, tempShort) == 1
&& idGrid->getData(i + 1, ny - 1, tempLong) >= 0) {
if (sendLink(idGrid->getData(i + 1, ny - 1, tempLong),
rank + 1)) {
linksSent++;
} else {
MPI_Abort(MCW, 2);
}
}
}
}
}
} else {
int i = 0;
if (rank != 0) {
MPI_Recv(&toRecv, 1, MPI_INT, rank - 1, rank - 1, MCW, &stat);
int i = 0;
for (i = 0; i < toRecv; i++) {
if (recvLink(rank - 1)) {
linksRecv++;
} else {
MPI_Abort(MCW, 2);
}
}
}
i = 0;
if (rank != size - 1) {
MPI_Recv(&toRecv, 1, MPI_INT, rank + 1, rank + 1, MCW, &stat);
int i = 0;
for (i = 0; i < toRecv; i++) {
if (recvLink(rank + 1)) {
linksRecv++;
} else {
MPI_Abort(MCW, 2);
}
}
}
}
MPI_Barrier(MCW);
MPI_Allreduce(&linksSent, &totalSent, 1, MPI_INT, MPI_SUM, MCW);
MPI_Allreduce(&linksRecv, &totalRecv, 1, MPI_INT, MPI_SUM, MCW);
int diff = totalSent - totalRecv;
if (diff != 0) {
MPI_Abort(MCW, 1);
}
return;
}
long getMagnitude(int32_t Id) {
streamlink *myLink = FindLink(Id);
if (myLink == NULL) {
MPI_Abort(MCW, 7);
}
return myLink->magnitude;
}
void getNumLinksAndPoints(long &NumLinks, long &NumPoints) {
long ID = -1;
llnode *current = linkSet.head;
NumLinks = linkSet.numLinks;
NumPoints = 0;
if (linkSet.numLinks == 0) {
return;
} else {
while (current != NULL) {
NumPoints += current->data->numCoords;
current = current->next;
}
}
return;
}

void setLinkInfo(long **LinkIdU1U2DMagShapeid,
double **LinkElevUElevDLength,
double **PointXY,
float **PointElevArea,
tdpartition *elev,
tiffIO *elevIO) {
long counter = 0;
long pointsSoFar = 0;
llnode *current = linkSet.head;
if (linkSet.numLinks == 0) {
return;
} else {
long begcoord = 0;
while (current != NULL) {
LinkIdU1U2DMagShapeid[counter][0] = current->data->Id;
LinkIdU1U2DMagShapeid[counter][1] = begcoord;
LinkIdU1U2DMagShapeid[counter][2] = begcoord + current->data->numCoords - 1;
begcoord = LinkIdU1U2DMagShapeid[counter][2] + 1;
LinkIdU1U2DMagShapeid[counter][3] = current->data->d;
LinkIdU1U2DMagShapeid[counter][4] = current->data->u1;
LinkIdU1U2DMagShapeid[counter][5] = current->data->u2;
LinkIdU1U2DMagShapeid[counter][6] = current->data->order;
LinkIdU1U2DMagShapeid[counter][7] = current->data->shapeId;
LinkIdU1U2DMagShapeid[counter][8] = current->data->magnitude;

long i = 0;
double cellarea = (elev->getdxA()) * (elev->getdyA());
for (i = 0; i < current->data->numCoords; i++) {
elevIO->globalXYToGeo(current->data->coord.front().x,
current->data->coord.front().y,
PointXY[pointsSoFar][0],
PointXY[pointsSoFar][1]);
PointElevArea[pointsSoFar][0] = current->data->coord.front().length;
PointElevArea[pointsSoFar][1] = current->data->coord.front().elev;
PointElevArea[pointsSoFar][2] = current->data->coord.front().area * cellarea;
current->data->coord.pop();
pointsSoFar++;
}
current = current->next;
counter++;
}
}
return;
}
streamlink *getFirstLink() {
streamlink *first;
first = linkSet.head->data;
return first;

}
streamlink *getLink(long node) {
llnode *current = linkSet.head;
if (current == NULL) {
return NULL;
}
long i = 0;
while (current != NULL) {
if (i == node) {
return current->data;
}
current = current->next;
i++;
}
return NULL;
}

point *initPoint(tdpartition *elev, tdpartition *areaD8, tdpartition *lengths, long i, long j) {
point *thePoint;
thePoint = new point[1];
int gi, gj;
float tempFloat;
elev->localToGlobal((int) i, (int) j, gi, gj);
thePoint->x = gi;
thePoint->y = gj;
thePoint->elev = elev->getData(i, j, tempFloat);
thePoint->area = areaD8->getData(i, j, tempFloat);
thePoint->length = lengths->getData(i, j, tempFloat);
return (thePoint);
}

bool ReceiveWaitingLinks(int rank, int size) {
if (size <= 1)return (false);  
int flag, flagup, flagdown, temp;
MPI_Status status;
MPI_Iprobe(MPI_ANY_SOURCE, 10, MCW, &flag, &status);
while (flag == 1) {
recvLink(status.MPI_SOURCE);
MPI_Iprobe(MPI_ANY_SOURCE, 10, MCW, &flag, &status);
}
if (rank == 0) {
MPI_Iprobe(rank + 1, 66, MCW, &flag, &status);
if (flag == 1) {
MPI_Recv(&temp, 1, MPI_INT, rank + 1, 66, MCW, &status);
return (false);
}
return (true);
}
if (rank == size - 1) {
MPI_Iprobe(rank - 1, 66, MCW, &flag, &status);
if (flag == 1) {
MPI_Recv(&temp, 1, MPI_INT, rank - 1, 66, MCW, &status);
return (false);
}
return (true);
}
MPI_Iprobe(rank - 1, 66, MCW, &flagup, &status);
MPI_Iprobe(rank + 1, 66, MCW, &flagdown, &status);
if ((flagup == 1) && (flagdown == 1)) {
MPI_Recv(&temp, 1, MPI_INT, rank - 1, 66, MCW, &status);
MPI_Recv(&temp, 1, MPI_INT, rank + 1, 66, MCW, &status);
return (false);
}
return (true);
}
