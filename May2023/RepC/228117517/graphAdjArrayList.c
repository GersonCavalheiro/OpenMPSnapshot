#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <stdint.h>
#include <omp.h>
#include "timer.h"
#include "myMalloc.h"
#include "graphConfig.h"
#include "edgeList.h"
#include "sortRun.h"
#include "vertex.h"
#include "adjArrayList.h"
#include "graphAdjArrayList.h"
#include "reorder.h"
void graphAdjArrayListPrintMessageWithtime(const char *msg, double time)
{
printf(" -----------------------------------------------------\n");
printf("| %-51s | \n", msg);
printf(" -----------------------------------------------------\n");
printf("| %-51f | \n", time);
printf(" -----------------------------------------------------\n");
}
struct GraphAdjArrayList *graphAdjArrayListGraphNew(uint32_t V)
{
struct GraphAdjArrayList *graphAdjArrayList = (struct GraphAdjArrayList *) my_malloc( sizeof(struct GraphAdjArrayList));
graphAdjArrayList->num_vertices = V;
graphAdjArrayList->vertices = (struct AdjArrayList *) my_malloc( V * sizeof(struct AdjArrayList));
uint32_t i;
for(i = 0; i < V; i++)
{
graphAdjArrayList->vertices[i].outNodes = NULL;
graphAdjArrayList->vertices[i].out_degree = 0;
#if DIRECTED
graphAdjArrayList->vertices[i].inNodes = NULL;
graphAdjArrayList->vertices[i].in_degree = 0;
#endif
}
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgeListNew(struct EdgeList *edgeList)
{
struct Timer *timer = (struct Timer *) my_malloc( sizeof(struct Timer));
struct GraphAdjArrayList *graphAdjArrayList;
Start(timer);
graphAdjArrayList = graphAdjArrayListGraphNew(edgeList->num_vertices);
graphAdjArrayList->num_edges = edgeList->num_edges;
graphAdjArrayList->avg_degree = edgeList->num_edges / edgeList->num_vertices;
#if WEIGHTED
graphAdjArrayList->max_weight =  edgeList->max_weight;
#endif
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList New (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgeListProcessOutDegree(graphAdjArrayList, edgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Process OutDegree (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgeAllocateOutNodes(graphAdjArrayList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Allocate Memory (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgePopulateOutNodes(graphAdjArrayList, edgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Populate OutNodes (Seconds)", Seconds(timer));
freeEdgeList(edgeList);
free(timer);
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgeListNewWithInverse(struct EdgeList *edgeList, struct EdgeList *inverseEdgeList)
{
struct Timer *timer = (struct Timer *) my_malloc( sizeof(struct Timer));
struct GraphAdjArrayList *graphAdjArrayList;
Start(timer);
graphAdjArrayList = graphAdjArrayListGraphNew(edgeList->num_vertices);
graphAdjArrayList->num_edges = edgeList->num_edges;
graphAdjArrayList->avg_degree = edgeList->num_edges / edgeList->num_vertices;
#if WEIGHTED
graphAdjArrayList->max_weight =  edgeList->max_weight;
#endif
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList New (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgeListProcessOutDegree(graphAdjArrayList, edgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Process OutDegree (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgeAllocateOutNodes(graphAdjArrayList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Allocate Memory (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgePopulateOutNodes(graphAdjArrayList, edgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Populate OutNodes (Seconds)", Seconds(timer));
freeEdgeList(edgeList);
#if DIRECTED
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgeListProcessInDegree(graphAdjArrayList, inverseEdgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Process InDegree (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgeAllocateInodes(graphAdjArrayList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Allocate Memory (Seconds)", Seconds(timer));
Start(timer);
graphAdjArrayList = graphAdjArrayListEdgePopulateInNodes(graphAdjArrayList, inverseEdgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Graph AdjArrayList Populate InNodes (Seconds)", Seconds(timer));
freeEdgeList(inverseEdgeList);
#endif
free(timer);
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgeListProcessInOutDegree(struct GraphAdjArrayList *graphAdjArrayList, struct EdgeList *edgeList)
{
uint32_t i;
uint32_t src;
#if DIRECTED
uint32_t dest;
#endif
#pragma omp parallel for
for(i = 0; i < edgeList->num_edges; i++)
{
src =  edgeList->edges_array_src[i];
#pragma omp atomic update
graphAdjArrayList->vertices[src].out_degree++;
#if DIRECTED
dest = edgeList->edges_array_dest[i];
#pragma omp atomic update
graphAdjArrayList->vertices[dest].in_degree++;
#endif
}
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgeListProcessOutDegree(struct GraphAdjArrayList *graphAdjArrayList, struct EdgeList *edgeList)
{
uint32_t i;
uint32_t src;
#pragma omp parallel for
for(i = 0; i < edgeList->num_edges; i++)
{
src =  edgeList->edges_array_src[i];
#pragma omp atomic update
graphAdjArrayList->vertices[src].out_degree++;
}
return graphAdjArrayList;
}
#if DIRECTED
struct GraphAdjArrayList *graphAdjArrayListEdgeListProcessInDegree(struct GraphAdjArrayList *graphAdjArrayList, struct EdgeList *inverseEdgeList)
{
uint32_t i;
uint32_t dest;
#pragma omp parallel for
for(i = 0; i < inverseEdgeList->num_edges; i++)
{
dest =  inverseEdgeList->edges_array_src[i];
#pragma omp atomic update
graphAdjArrayList->vertices[dest].in_degree++;
}
return graphAdjArrayList;
}
#endif
struct GraphAdjArrayList *graphAdjArrayListEdgeAllocate(struct GraphAdjArrayList *graphAdjArrayList)
{
uint32_t v;
#pragma omp parallel for
for(v = 0; v < graphAdjArrayList->num_vertices; v++)
{
adjArrayListCreateNeighbourList(&(graphAdjArrayList->vertices[v]));
#if DIRECTED
graphAdjArrayList->vertices[v].in_degree =  0;
#endif
graphAdjArrayList->vertices[v].out_degree = 0; 
}
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgeAllocateInodes(struct GraphAdjArrayList *graphAdjArrayList)
{
#if DIRECTED
uint32_t v;
for(v = 0; v < graphAdjArrayList->num_vertices; v++)
{
adjArrayListCreateNeighbourListInNodes(&(graphAdjArrayList->vertices[v]));
graphAdjArrayList->vertices[v].in_degree =  0;
}
#endif
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgeAllocateOutNodes(struct GraphAdjArrayList *graphAdjArrayList)
{
uint32_t v;
for(v = 0; v < graphAdjArrayList->num_vertices; v++)
{
adjArrayListCreateNeighbourListOutNodes(&(graphAdjArrayList->vertices[v]));
graphAdjArrayList->vertices[v].out_degree = 0; 
}
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgePopulate(struct GraphAdjArrayList *graphAdjArrayList, struct EdgeList *edgeList)
{
uint32_t i;
uint32_t src;
#if DIRECTED
uint32_t dest;
uint32_t in_degree;
#endif
uint32_t out_degree;
for(i = 0; i < edgeList->num_edges; i++)
{
src =  edgeList->edges_array_src[i];
out_degree = graphAdjArrayList->vertices[src].out_degree;
graphAdjArrayList->vertices[src].outNodes->edges_array_src[out_degree] = edgeList->edges_array_src[i];
graphAdjArrayList->vertices[src].outNodes->edges_array_dest[out_degree] = edgeList->edges_array_dest[i];
#if WEIGHTED
graphAdjArrayList->vertices[src].outNodes->edges_array_weight[out_degree] = edgeList->edges_array_weight[i];
#endif
graphAdjArrayList->vertices[src].out_degree++;
#if DIRECTED
dest = edgeList->edges_array_dest[i];
in_degree = __sync_fetch_and_add(&(graphAdjArrayList->vertices[src].in_degree), 1);
graphAdjArrayList->vertices[dest].inNodes->edges_array_src[in_degree] = edgeList->edges_array_src[i];
graphAdjArrayList->vertices[dest].inNodes->edges_array_dest[in_degree] = edgeList->edges_array_dest[i];
#if WEIGHTED
graphAdjArrayList->vertices[dest].inNodes->edges_array_weight[in_degree] = edgeList->edges_array_weight[i];
#endif
#endif
}
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgePopulateOutNodes(struct GraphAdjArrayList *graphAdjArrayList, struct EdgeList *edgeList)
{
uint32_t i;
uint32_t src;
uint32_t out_degree;
for(i = 0; i < edgeList->num_edges; i++)
{
src =  edgeList->edges_array_src[i];
out_degree = graphAdjArrayList->vertices[src].out_degree;
graphAdjArrayList->vertices[src].outNodes->edges_array_src[out_degree] = edgeList->edges_array_src[i];
graphAdjArrayList->vertices[src].outNodes->edges_array_dest[out_degree] = edgeList->edges_array_dest[i];
#if WEIGHTED
graphAdjArrayList->vertices[src].outNodes->edges_array_weight[out_degree] = edgeList->edges_array_weight[i];
#endif
graphAdjArrayList->vertices[src].out_degree++;
}
return graphAdjArrayList;
}
struct GraphAdjArrayList *graphAdjArrayListEdgePopulateInNodes(struct GraphAdjArrayList *graphAdjArrayList, struct EdgeList *inverseEdgeList)
{
#if DIRECTED
uint32_t i;
uint32_t dest;
uint32_t in_degree;
for(i = 0; i < inverseEdgeList->num_edges; i++)
{
dest = inverseEdgeList->edges_array_src[i];
in_degree = graphAdjArrayList->vertices[dest].in_degree;
graphAdjArrayList->vertices[dest].inNodes->edges_array_src[in_degree] = inverseEdgeList->edges_array_src[i];
graphAdjArrayList->vertices[dest].inNodes->edges_array_dest[in_degree] = inverseEdgeList->edges_array_dest[i];
#if WEIGHTED
graphAdjArrayList->vertices[dest].inNodes->edges_array_weight[in_degree] = inverseEdgeList->edges_array_weight[i];
#endif
graphAdjArrayList->vertices[dest].in_degree++;
}
#endif
return graphAdjArrayList;
}
void graphAdjArrayListPrint(struct GraphAdjArrayList *graphAdjArrayList)
{
printf(" -----------------------------------------------------\n");
printf("| %-51s | \n", "GraphAdjArrayList Properties");
printf(" -----------------------------------------------------\n");
#if WEIGHTED
printf("| %-51s | \n", "WEIGHTED");
#else
printf("| %-51s | \n", "UN-WEIGHTED");
#endif
#if DIRECTED
printf("| %-51s | \n", "DIRECTED");
#else
printf("| %-51s | \n", "UN-DIRECTED");
#endif
printf(" -----------------------------------------------------\n");
printf("| %-51s | \n", "Average Degree (D)");
printf("| %-51u | \n", graphAdjArrayList->avg_degree);
printf(" -----------------------------------------------------\n");
printf("| %-51s | \n", "Number of Vertices (V)");
printf("| %-51u | \n", graphAdjArrayList->num_vertices);
printf(" -----------------------------------------------------\n");
printf("| %-51s | \n", "Number of Edges (E)");
printf("| %-51u | \n", graphAdjArrayList->num_edges);
printf(" -----------------------------------------------------\n");
}
void graphAdjArrayListFree(struct GraphAdjArrayList *graphAdjArrayList)
{
uint32_t v;
struct AdjArrayList *pCrawl;
for (v = 0; v < graphAdjArrayList->num_vertices; ++v)
{
pCrawl = &(graphAdjArrayList->vertices[v]);
if(pCrawl->outNodes)
freeEdgeList(pCrawl->outNodes);
#if DIRECTED
if(pCrawl->inNodes)
freeEdgeList(pCrawl->inNodes);
#endif
}
if(graphAdjArrayList->vertices)
free(graphAdjArrayList->vertices);
if(graphAdjArrayList)
free(graphAdjArrayList);
}
struct GraphAdjArrayList *graphAdjArrayListPreProcessingStep (struct Arguments *arguments)
{
struct Timer *timer = (struct Timer *) my_malloc(sizeof(struct Timer));
Start(timer);
struct EdgeList *edgeList = readEdgeListsbin(arguments->fnameb, 0, arguments->symmetric, arguments->weighted);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Read Edge List From File (Seconds)", Seconds(timer));
edgeList = sortRunAlgorithms(edgeList, arguments->sort);
if(arguments->dflag)
{
Start(timer);
edgeList = removeDulpicatesSelfLoopEdges(edgeList);
Stop(timer);
graphCSRPrintMessageWithtime("Removing duplicate edges (Seconds)", Seconds(timer));
}
if(arguments->lmode)
{
edgeList = reorderGraphProcess(edgeList, arguments);
edgeList = sortRunAlgorithms(edgeList, arguments->sort);
}
arguments->lmode = arguments->lmode_l2;
if(arguments->lmode)
{
edgeList = reorderGraphProcess(edgeList, arguments);
edgeList = sortRunAlgorithms(edgeList, arguments->sort);
}
arguments->lmode = arguments->lmode_l3;
if(arguments->lmode)
{
edgeList = reorderGraphProcess(edgeList, arguments);
edgeList = sortRunAlgorithms(edgeList, arguments->sort);
}
if(arguments->mmode)
edgeList = maskGraphProcess(edgeList, arguments);
#if DIRECTED
Start(timer);
struct EdgeList *inverse_edgeList = readEdgeListsMem(edgeList, 1, 0, 0);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Read Inverse Edge List From File (Seconds)", Seconds(timer));
inverse_edgeList = sortRunAlgorithms(inverse_edgeList, arguments->sort);
#endif
#if DIRECTED
Start(timer);
struct GraphAdjArrayList *graph = graphAdjArrayListEdgeListNewWithInverse(edgeList, inverse_edgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Create AdjArrayList from EdgeList (Seconds)", Seconds(timer));
#else
Start(timer);
struct GraphAdjArrayList *graph = graphAdjArrayListEdgeListNew(edgeList);
Stop(timer);
graphAdjArrayListPrintMessageWithtime("Create AdjArrayList from EdgeList (Seconds)", Seconds(timer));
#endif
graphAdjArrayListPrint(graph);
free(timer);
return graph;
}
