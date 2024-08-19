

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>
#include <cassert>
#include <inttypes.h>
#include <fstream>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include "common.h"
typedef unsigned foru;

#include "graph.h"
#include "component.h"
#include "devel.h"

int num_omp_threads;

void init(Graph graph, ComponentSpace cs, foru *eleminwts, 
foru *minwtcomponent, unsigned *partners, 
bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
#ifdef TIMING
double starttime = rtclock();
#endif
#pragma omp parallel for schedule(static)
for (int id=0; id < graph.nnodes; id++) {
eleminwts[id] = MYINFINITY;
minwtcomponent[id] = MYINFINITY;
goaheadnodeofcomponent[id] = graph.nnodes;
partners[id] = id;
processinnextiteration[id] = false;
}
#ifdef TIMING
double endtime = rtclock();
printf("\truntime [init] = %f ms.\n", 1000 * (endtime - starttime));
#endif
}

void findelemin(Graph graph, ComponentSpace cs, foru *eleminwts, 
foru *minwtcomponent, unsigned *partners, 
bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
#ifdef TIMING
double starttime = rtclock();
#endif
#pragma omp parallel for schedule(static)
for(int id = 0; id < graph.nnodes; id ++) {
unsigned src = id;
unsigned srcboss = cs.find(src);
unsigned dstboss = graph.nnodes;
foru minwt = MYINFINITY;
unsigned degree = graph.getOutDegree(src);
for (unsigned ii = 0; ii < degree; ++ii) {
foru wt = graph.getWeight(src, ii);
if (wt < minwt) {
unsigned dst = graph.getDestination(src, ii);
unsigned tempdstboss = cs.find(dst);
if (srcboss != tempdstboss) {
minwt = wt;
dstboss = tempdstboss;
}
}
}
eleminwts[id] = minwt;
partners[id] = dstboss;

if (minwt < minwtcomponent[srcboss] && srcboss != dstboss) {
#pragma omp critical
{
if (minwt < minwtcomponent[srcboss])
minwtcomponent[srcboss] = minwt;
}
}
}
#ifdef TIMING
double endtime = rtclock();
printf("\truntime [findelemin] = %f ms.\n", 1000 * (endtime - starttime));
#endif
}

void findelemin2(Graph graph, ComponentSpace cs, foru *eleminwts, 
foru *minwtcomponent, unsigned *partners, 
bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
#ifdef TIMING
double starttime = rtclock();
#endif
#pragma omp parallel for schedule(static)
for (int id=0; id < graph.nnodes; id++) {
unsigned src = id;
unsigned srcboss = cs.find(src);
if(eleminwts[id] == minwtcomponent[srcboss] && 
srcboss != partners[id] && partners[id] != graph.nnodes) {
unsigned degree = graph.getOutDegree(src);
for (unsigned ii = 0; ii < degree; ++ii) {
foru wt = graph.getWeight(src, ii);
if (wt == eleminwts[id]) {
unsigned dst = graph.getDestination(src, ii);
unsigned tempdstboss = cs.find(dst);
if (tempdstboss == partners[id]) {
my_compare_swap<unsigned>(&goaheadnodeofcomponent[srcboss], graph.nnodes, id);
}
}
}
}
}
#ifdef TIMING
double endtime = rtclock();
printf("\truntime [findelemin2] = %f ms.\n", 1000 * (endtime - starttime));
#endif
}

void verify_min_elem(Graph graph, ComponentSpace cs, foru *eleminwts, 
foru *minwtcomponent, unsigned *partners, 
bool *processinnextiteration, unsigned *goaheadnodeofcomponent) {
#ifdef TIMING
double starttime = rtclock();
#endif
#pragma omp parallel for schedule(static)
for (int id=0; id < graph.nnodes; id++) {
if(cs.isBoss(id)) {
if(goaheadnodeofcomponent[id] == graph.nnodes) {
continue;
}
unsigned minwt_node = goaheadnodeofcomponent[id];
unsigned degree = graph.getOutDegree(minwt_node);
foru minwt = minwtcomponent[id];
if(minwt == MYINFINITY)
continue;
bool minwt_found = false;
for (unsigned ii = 0; ii < degree; ++ii) {
foru wt = graph.getWeight(minwt_node, ii);

if (wt == minwt) {
minwt_found = true;
unsigned dst = graph.getDestination(minwt_node, ii);
unsigned tempdstboss = cs.find(dst);
if(tempdstboss == partners[minwt_node] && tempdstboss != id) {
processinnextiteration[minwt_node] = true;
break;
}
}
}
assert(minwt_found);
}
}
#ifdef TIMING
double endtime = rtclock();
printf("\truntime [verify] = %f ms.\n", 1000 * (endtime - starttime));
#endif
}

void findcompmintwo(unsigned *mstwt, Graph graph, ComponentSpace &csw, 
foru *eleminwts, foru *minwtcomponent, unsigned *partners, 
bool *processinnextiteration, unsigned *goaheadnodeofcomponent, 
bool *repeat, unsigned *count) {
#ifdef TIMING
double starttime = rtclock();
#endif
unsigned up = graph.nnodes;
#ifdef ENABLE_OPENMP
up = (up+num_omp_threads-1)/num_omp_threads*num_omp_threads;
#endif
for(int id = 0; id < up; id ++) {
unsigned srcboss, dstboss;
if(id < graph.nnodes && processinnextiteration[id]) {
srcboss = csw.find(id);
dstboss = csw.find(partners[id]);
}
__syncthreads();
if(id < graph.nnodes && processinnextiteration[id] && srcboss != dstboss) {
if (csw.unify(srcboss, dstboss)) {
#ifdef ENABLE_OPENMP
my_fetch_add<unsigned>(mstwt, eleminwts[id]);
#else
*mstwt += eleminwts[id];
#endif
#ifdef ENABLE_OPENMP
my_fetch_add<unsigned>(count, 1);
#else
(*count) ++;
#endif
processinnextiteration[id] = false;
eleminwts[id] = MYINFINITY;	
}
else {
*repeat = true;
}
}
__syncthreads();
}
#ifdef TIMING
double endtime = rtclock();
printf("\truntime [findcompmin] = %f ms.\n", 1000 * (endtime - starttime));
#endif
}

int main(int argc, char *argv[]) {
unsigned mstwt = 0;
int iteration = 0;
Graph graph;
unsigned *partners;
foru *eleminwts, *minwtcomponent;
bool *processinnextiteration;
unsigned *goaheadnodeofcomponent;
double starttime, endtime;
if (argc != 3) {
printf("Usage: %s <nThreads> <graph>\n", argv[0]);
exit(1);
}
num_omp_threads = atoi(argv[1]);
graph.read(argv[2]);
ComponentSpace cs(graph.nnodes);

eleminwts = (foru *)malloc(graph.nnodes * sizeof(foru));
minwtcomponent = (foru *)malloc(graph.nnodes * sizeof(foru));
partners = (unsigned *)malloc(graph.nnodes * sizeof(unsigned));
processinnextiteration = (bool *)malloc(graph.nnodes * sizeof(bool));
goaheadnodeofcomponent = (unsigned *)malloc(graph.nnodes * sizeof(unsigned));

unsigned prevncomponents, currncomponents = graph.nnodes;

bool repeat = false;
unsigned edgecount = 0;
#ifdef ENABLE_OPENMP
omp_set_num_threads(num_omp_threads);
#endif
printf("finding mst.\n");
starttime = rtclock();
do {
++iteration;
prevncomponents = currncomponents;
init(graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
findelemin(graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
findelemin2(graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
verify_min_elem(graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent);
if(0) print_comp_mins(cs, graph, minwtcomponent, goaheadnodeofcomponent, partners, processinnextiteration);
do {
repeat = false;
findcompmintwo(&mstwt, graph, cs, eleminwts, minwtcomponent, partners, processinnextiteration, goaheadnodeofcomponent, &repeat, &edgecount);
} while (repeat); 
currncomponents = cs.numberOfComponents();
printf("\titeration %d, number of components = %d (%d), mstwt = %u mstedges = %u\n", iteration, currncomponents, prevncomponents, mstwt, edgecount);
} while (currncomponents != prevncomponents);

endtime = rtclock();
printf("\tmstwt = %u, iterations = %d.\n", mstwt, iteration);
printf("\t%s result: weight: %u, components: %u, edges: %u\n", argv[2], mstwt, currncomponents, edgecount);
printf("\truntime [mst] = %f ms.\n", 1000 * (endtime - starttime));

cs.deallocate();
free(eleminwts);
free(minwtcomponent);
free(partners);
free(processinnextiteration);
free(goaheadnodeofcomponent);
return 0;
}
