





#include "kdtree.h"
#include "generic.h"
#include "random.h"
#include "mathop.h"
#include <stdlib.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#define VL_HEAP_prefix     vl_kdforest_search_heap
#define VL_HEAP_type       VlKDForestSearchState
#define VL_HEAP_cmp(v,x,y) (v[x].distanceLowerBound - v[y].distanceLowerBound)
#include "heap-def.h"

#define VL_HEAP_prefix     vl_kdtree_split_heap
#define VL_HEAP_type       VlKDTreeSplitDimension
#define VL_HEAP_cmp(v,x,y) (v[x].variance - v[y].variance)
#include "heap-def.h"

#define VL_HEAP_prefix     vl_kdforest_neighbor_heap
#define VL_HEAP_type       VlKDForestNeighbor
#define VL_HEAP_cmp(v,x,y) (v[y].distance - v[x].distance)
#include "heap-def.h"



static vl_uindex
vl_kdtree_node_new (VlKDTree * tree, vl_uindex parentIndex)
{
VlKDTreeNode * node = NULL ;
vl_uindex nodeIndex = tree->numUsedNodes ;
tree -> numUsedNodes += 1 ;

assert (tree->numUsedNodes <= tree->numAllocatedNodes) ;

node = tree->nodes + nodeIndex ;
node -> parent = parentIndex ;
node -> lowerChild = 0 ;
node -> upperChild = 0 ;
node -> splitDimension = 0 ;
node -> splitThreshold = 0 ;
return nodeIndex ;
}



VL_INLINE int
vl_kdtree_compare_index_entries (void const * a,
void const * b)
{
double delta =
((VlKDTreeDataIndexEntry const*)a) -> value -
((VlKDTreeDataIndexEntry const*)b) -> value ;
if (delta < 0) return -1 ;
if (delta > 0) return +1 ;
return 0 ;
}



static void
vl_kdtree_build_recursively
(VlKDForest * forest,
VlKDTree * tree, vl_uindex nodeIndex,
vl_uindex dataBegin, vl_uindex dataEnd,
unsigned int depth)
{
vl_uindex d, i, medianIndex, splitIndex ;
VlKDTreeNode * node = tree->nodes + nodeIndex ;
VlKDTreeSplitDimension * splitDimension ;


if (dataEnd - dataBegin <= 1) {
if (tree->depth < depth) tree->depth = depth ;
node->lowerChild = - dataBegin - 1;
node->upperChild = - dataEnd - 1 ;
return ;
}


forest->splitHeapNumNodes = 0 ;
for (d = 0 ; d < forest->dimension ; ++ d) {
double mean = 0 ; 
double secondMoment = 0 ;
double variance = 0 ;
vl_size numSamples = VL_KDTREE_VARIANCE_EST_NUM_SAMPLES;
vl_bool useAllData = VL_FALSE;

if(dataEnd - dataBegin <= VL_KDTREE_VARIANCE_EST_NUM_SAMPLES) {
useAllData = VL_TRUE;
numSamples = dataEnd - dataBegin;
}

for (i = 0; i < numSamples ; ++ i) {
vl_uint32 sampleIndex;
vl_index di;
double datum ;

if(useAllData == VL_TRUE) {
sampleIndex = (vl_uint32)i;
} else {
sampleIndex = (vl_rand_uint32(forest->rand) % VL_KDTREE_VARIANCE_EST_NUM_SAMPLES);
}
sampleIndex += dataBegin;

di = tree->dataIndex[sampleIndex].index ;

switch(forest->dataType) {
case VL_TYPE_FLOAT: datum = ((float const*)forest->data)
[di * forest->dimension + d] ;
break ;
case VL_TYPE_DOUBLE: datum = ((double const*)forest->data)
[di * forest->dimension + d] ;
break ;
default:
abort() ;
}
mean += datum ;
secondMoment += datum * datum ;
}

mean /= numSamples ;
secondMoment /= numSamples ;
variance = secondMoment - mean * mean ;

if (variance <= 0) continue ;


if (forest->splitHeapNumNodes < forest->splitHeapSize) {
VlKDTreeSplitDimension * splitDimension
= forest->splitHeapArray + forest->splitHeapNumNodes ;
splitDimension->dimension = (unsigned int)d ;
splitDimension->mean = mean ;
splitDimension->variance = variance ;
vl_kdtree_split_heap_push (forest->splitHeapArray, &forest->splitHeapNumNodes) ;
} else {
VlKDTreeSplitDimension * splitDimension = forest->splitHeapArray + 0 ;
if (splitDimension->variance < variance) {
splitDimension->dimension = (unsigned int)d ;
splitDimension->mean = mean ;
splitDimension->variance = variance ;
vl_kdtree_split_heap_update (forest->splitHeapArray, forest->splitHeapNumNodes, 0) ;
}
}
}


if (forest->splitHeapNumNodes == 0) {
node->lowerChild = - dataBegin - 1 ;
node->upperChild = - dataEnd - 1 ;
return ;
}


splitDimension = forest->splitHeapArray
+ (vl_rand_uint32(forest->rand) % VL_MIN(forest->splitHeapSize, forest->splitHeapNumNodes)) ;

node->splitDimension = splitDimension->dimension ;


for (i = dataBegin ; i < dataEnd ; ++ i) {
vl_index di = tree->dataIndex[i].index ;
double datum ;
switch (forest->dataType) {
case VL_TYPE_FLOAT: datum = ((float const*)forest->data)
[di * forest->dimension + splitDimension->dimension] ;
break ;
case VL_TYPE_DOUBLE: datum = ((double const*)forest->data)
[di * forest->dimension + splitDimension->dimension] ;
break ;
default:
abort() ;
}
tree->dataIndex [i] .value = datum ;
}
qsort (tree->dataIndex + dataBegin,
dataEnd - dataBegin,
sizeof (VlKDTreeDataIndexEntry),
vl_kdtree_compare_index_entries) ;


switch (forest->thresholdingMethod) {
case VL_KDTREE_MEAN :
node->splitThreshold = splitDimension->mean ;
for (splitIndex = dataBegin ;
splitIndex < dataEnd && tree->dataIndex[splitIndex].value <= node->splitThreshold ;
++ splitIndex) ;
splitIndex -= 1 ;

if (dataBegin <= splitIndex && splitIndex + 1 < dataEnd) break ;

case VL_KDTREE_MEDIAN :
medianIndex = (dataBegin + dataEnd - 1) / 2 ;
splitIndex = medianIndex ;
node -> splitThreshold = tree->dataIndex[medianIndex].value ;
break ;

default:
abort() ;
}


node->lowerChild = vl_kdtree_node_new (tree, nodeIndex) ;
vl_kdtree_build_recursively (forest, tree, node->lowerChild, dataBegin, splitIndex + 1, depth + 1) ;

node->upperChild = vl_kdtree_node_new (tree, nodeIndex) ;
vl_kdtree_build_recursively (forest, tree, node->upperChild, splitIndex + 1, dataEnd, depth + 1) ;
}



VlKDForest *
vl_kdforest_new (vl_type dataType,
vl_size dimension, vl_size numTrees, VlVectorComparisonType distance)
{
VlKDForest * self = vl_calloc (sizeof(VlKDForest), 1) ;

assert(dataType == VL_TYPE_FLOAT || dataType == VL_TYPE_DOUBLE) ;
assert(dimension >= 1) ;
assert(numTrees >= 1) ;

self -> rand = vl_get_rand () ;
self -> dataType = dataType ;
self -> numData = 0 ;
self -> data = 0 ;
self -> dimension = dimension ;
self -> numTrees = numTrees ;
self -> trees = 0 ;
self -> thresholdingMethod = VL_KDTREE_MEDIAN ;
self -> splitHeapSize = VL_MIN(numTrees, VL_KDTREE_SPLIT_HEAP_SIZE) ;
self -> splitHeapNumNodes = 0 ;
self -> distance = distance;
self -> maxNumNodes = 0 ;
self -> numSearchers = 0 ;
self -> headSearcher = 0 ;

switch (self->dataType) {
case VL_TYPE_FLOAT:
self -> distanceFunction = (void(*)(void))
vl_get_vector_comparison_function_f (distance) ;
break;
case VL_TYPE_DOUBLE :
self -> distanceFunction = (void(*)(void))
vl_get_vector_comparison_function_d (distance) ;
break ;
default :
abort() ;
}

return self ;
}



VlKDForestSearcher *
vl_kdforest_new_searcher (VlKDForest * kdforest)
{
VlKDForestSearcher * self = vl_calloc(sizeof(VlKDForestSearcher), 1);
if(kdforest->numSearchers == 0) {
kdforest->headSearcher = self;
self->previous = NULL;
self->next = NULL;
} else {
VlKDForestSearcher * lastSearcher = kdforest->headSearcher;
while (1) {
if(lastSearcher->next) {
lastSearcher = lastSearcher->next;
} else {
lastSearcher->next = self;
self->previous = lastSearcher;
self->next = NULL;
break;
}
}
}

kdforest->numSearchers++;

self->forest = kdforest;
self->searchHeapArray = vl_malloc (sizeof(VlKDForestSearchState) * kdforest->maxNumNodes) ;
self->searchIdBook = vl_calloc (sizeof(vl_uindex), kdforest->numData) ;
return self ;
}



void
vl_kdforestsearcher_delete (VlKDForestSearcher * self)
{
if (self->previous && self->next) {
self->previous->next = self->next;
self->next->previous = self->previous;
} else if (self->previous && !self->next) {
self->previous->next = NULL;
} else if (!self->previous && self->next) {
self->next->previous = NULL;
self->forest->headSearcher = self->next;
} else {
self->forest->headSearcher = NULL;
}
self->forest->numSearchers -- ;
vl_free(self->searchHeapArray) ;
vl_free(self->searchIdBook) ;
vl_free(self) ;
}

VlKDForestSearcher *
vl_kdforest_get_searcher (VlKDForest const * self, vl_uindex pos)
{
VlKDForestSearcher * lastSearcher = self->headSearcher ;
vl_uindex i ;

for(i = 0; (i < pos) & (lastSearcher != NULL) ; ++i) {
lastSearcher = lastSearcher->next ;
}
return lastSearcher ;
}



void
vl_kdforest_delete (VlKDForest * self)
{
vl_uindex ti ;
VlKDForestSearcher * searcher ;

while ((searcher = vl_kdforest_get_searcher(self, 0))) {
vl_kdforestsearcher_delete(searcher) ;
}

if (self->trees) {
for (ti = 0 ; ti < self->numTrees ; ++ ti) {
if (self->trees[ti]) {
if (self->trees[ti]->nodes) vl_free (self->trees[ti]->nodes) ;
if (self->trees[ti]->dataIndex) vl_free (self->trees[ti]->dataIndex) ;
vl_free (self->trees[ti]) ;
}
}
vl_free (self->trees) ;
}
vl_free (self) ;
}



static void
vl_kdtree_calc_bounds_recursively (VlKDTree * tree,
vl_uindex nodeIndex, double * searchBounds)
{
VlKDTreeNode * node = tree->nodes + nodeIndex ;
vl_uindex i = node->splitDimension ;
double t = node->splitThreshold ;

node->lowerBound = searchBounds [2 * i + 0] ;
node->upperBound = searchBounds [2 * i + 1] ;


if (node->lowerChild > 0) {
searchBounds [2 * i + 1] = t ;
vl_kdtree_calc_bounds_recursively (tree, node->lowerChild, searchBounds) ;
searchBounds [2 * i + 1] = node->upperBound ;
}
if (node->upperChild > 0) {
searchBounds [2 * i + 0] = t ;
vl_kdtree_calc_bounds_recursively (tree, node->upperChild, searchBounds) ;
searchBounds [2 * i + 0] = node->lowerBound ;
}
}



void
vl_kdforest_build (VlKDForest * self, vl_size numData, void const * data)
{
vl_uindex di, ti ;
vl_size maxNumNodes ;
double * searchBounds;


self->data = data ;
self->numData = numData ;
self->trees = vl_malloc (sizeof(VlKDTree*) * self->numTrees) ;
maxNumNodes = 0 ;

for (ti = 0 ; ti < self->numTrees ; ++ ti) {
self->trees[ti] = vl_malloc (sizeof(VlKDTree)) ;
self->trees[ti]->dataIndex = vl_malloc (sizeof(VlKDTreeDataIndexEntry) * self->numData) ;
for (di = 0 ; di < self->numData ; ++ di) {
self->trees[ti]->dataIndex[di].index = di ;
}
self->trees[ti]->numUsedNodes = 0 ;

self->trees[ti]->numAllocatedNodes = 2 * self->numData - 1 ;
self->trees[ti]->nodes = vl_malloc (sizeof(VlKDTreeNode) * self->trees[ti]->numAllocatedNodes) ;
self->trees[ti]->depth = 0 ;
vl_kdtree_build_recursively (self, self->trees[ti],
vl_kdtree_node_new(self->trees[ti], 0), 0,
self->numData, 0) ;
maxNumNodes += self->trees[ti]->numUsedNodes ;
}

searchBounds = vl_malloc(sizeof(double) * 2 * self->dimension);

for (ti = 0 ; ti < self->numTrees ; ++ ti) {
double * iter = searchBounds  ;
double * end = iter + 2 * self->dimension ;
while (iter < end) {
*iter++ = - VL_INFINITY_F ;
*iter++ = + VL_INFINITY_F ;
}

vl_kdtree_calc_bounds_recursively (self->trees[ti], 0, searchBounds) ;
}

vl_free(searchBounds);
self -> maxNumNodes = maxNumNodes;
}




vl_uindex
vl_kdforest_query_recursively (VlKDForestSearcher * searcher,
VlKDTree * tree,
vl_uindex nodeIndex,
VlKDForestNeighbor * neighbors,
vl_size numNeighbors,
vl_size * numAddedNeighbors,
double dist,
void const * query)
{

VlKDTreeNode const * node = tree->nodes + nodeIndex ;
vl_uindex i = node->splitDimension ;
vl_index nextChild, saveChild ;
double delta, saveDist ;
double x ;
double x1 = node->lowerBound ;
double x2 = node->splitThreshold ;
double x3 = node->upperBound ;
VlKDForestSearchState * searchState ;

searcher->searchNumRecursions ++ ;

switch (searcher->forest->dataType) {
case VL_TYPE_FLOAT :
x = ((float const*) query)[i] ;
break ;
case VL_TYPE_DOUBLE :
x = ((double const*) query)[i] ;
break ;
default :
abort() ;
}


if (node->lowerChild < 0) {

vl_index begin = - node->lowerChild - 1 ;
vl_index end   = - node->upperChild - 1 ;
vl_index iter ;

for (iter = begin ;
iter < end &&
(searcher->forest->searchMaxNumComparisons == 0 ||
searcher->searchNumComparisons < searcher->forest->searchMaxNumComparisons) ;
++ iter) {

vl_index di = tree->dataIndex [iter].index ;


if (searcher->searchIdBook[di] == searcher->searchId) continue ;
searcher->searchIdBook[di] = searcher->searchId ;


switch (searcher->forest->dataType) {
case VL_TYPE_FLOAT:
dist = ((VlFloatVectorComparisonFunction)searcher->forest->distanceFunction)
(searcher->forest->dimension,
((float const *)query),
((float const*)searcher->forest->data) + di * searcher->forest->dimension) ;
break ;
case VL_TYPE_DOUBLE:
dist = ((VlDoubleVectorComparisonFunction)searcher->forest->distanceFunction)
(searcher->forest->dimension,
((double const *)query),
((double const*)searcher->forest->data) + di * searcher->forest->dimension) ;
break ;
default:
abort() ;
}
searcher->searchNumComparisons += 1 ;


if (*numAddedNeighbors < numNeighbors) {
VlKDForestNeighbor * newNeighbor = neighbors + *numAddedNeighbors ;
newNeighbor->index = di ;
newNeighbor->distance = dist ;
vl_kdforest_neighbor_heap_push (neighbors, numAddedNeighbors) ;
} else {
VlKDForestNeighbor * largestNeighbor = neighbors + 0 ;
if (largestNeighbor->distance > dist) {
largestNeighbor->index = di ;
largestNeighbor->distance = dist ;
vl_kdforest_neighbor_heap_update (neighbors, *numAddedNeighbors, 0) ;
}
}
} 


return nodeIndex ;
}

#if 0
assert (x1 <= x2 && x2 <= x3) ;
assert (node->lowerChild >= 0) ;
assert (node->upperChild >= 0) ;
#endif



delta = x - x2 ;
saveDist = dist + delta*delta ;

if (x <= x2) {
nextChild = node->lowerChild ;
saveChild = node->upperChild ;
if (x <= x1) {
delta = x - x1 ;
saveDist -= delta*delta ;
}
} else {
nextChild = node->upperChild ;
saveChild = node->lowerChild ;
if (x > x3) {
delta = x - x3 ;
saveDist -= delta*delta ;
}
}

if (*numAddedNeighbors < numNeighbors || neighbors[0].distance > saveDist) {
searchState = searcher->searchHeapArray + searcher->searchHeapNumNodes ;
searchState->tree = tree ;
searchState->nodeIndex = saveChild ;
searchState->distanceLowerBound = saveDist ;
vl_kdforest_search_heap_push (searcher->searchHeapArray ,
&searcher->searchHeapNumNodes) ;
}

return vl_kdforest_query_recursively (searcher,
tree,
nextChild,
neighbors,
numNeighbors,
numAddedNeighbors,
dist,
query) ;
}



vl_size
vl_kdforest_query (VlKDForest * self,
VlKDForestNeighbor * neighbors,
vl_size numNeighbors,
void const * query)
{
VlKDForestSearcher * searcher = vl_kdforest_get_searcher(self, 0) ;
if (searcher == NULL) {
searcher = vl_kdforest_new_searcher(self) ;
}
return vl_kdforestsearcher_query(searcher,
neighbors,
numNeighbors,
query) ;
}



vl_size
vl_kdforestsearcher_query (VlKDForestSearcher * self,
VlKDForestNeighbor * neighbors,
vl_size numNeighbors,
void const * query)
{

vl_uindex i, ti ;
vl_bool exactSearch = self->forest->searchMaxNumComparisons == 0 ;

VlKDForestSearchState * searchState  ;
vl_size numAddedNeighbors = 0 ;

assert (neighbors) ;
assert (numNeighbors > 0) ;
assert (query) ;


self -> searchId += 1 ;
self -> searchNumRecursions = 0 ;

self->searchNumComparisons = 0 ;
self->searchNumSimplifications = 0 ;


self->searchHeapNumNodes = 0 ;
for (ti = 0 ; ti < self->forest->numTrees ; ++ ti) {
searchState = self->searchHeapArray + self->searchHeapNumNodes ;
searchState -> tree = self->forest->trees[ti] ;
searchState -> nodeIndex = 0 ;
searchState -> distanceLowerBound = 0 ;

vl_kdforest_search_heap_push (self->searchHeapArray, &self->searchHeapNumNodes) ;
}


while (exactSearch || self->searchNumComparisons < self->forest->searchMaxNumComparisons)
{

VlKDForestSearchState * searchState ;


if (self->searchHeapNumNodes == 0) {
break ;
}
searchState = self->searchHeapArray +
vl_kdforest_search_heap_pop (self->searchHeapArray, &self->searchHeapNumNodes) ;

if (numAddedNeighbors == numNeighbors &&
neighbors[0].distance < searchState->distanceLowerBound) {
self->searchNumSimplifications ++ ;
break ;
}
vl_kdforest_query_recursively (self,
searchState->tree,
searchState->nodeIndex,
neighbors,
numNeighbors,
&numAddedNeighbors,
searchState->distanceLowerBound,
query) ;
}


for (i = numAddedNeighbors ; i < numNeighbors ; ++ i) {
neighbors[i].index = -1 ;
neighbors[i].distance = VL_NAN_F ;
}

while (numAddedNeighbors) {
vl_kdforest_neighbor_heap_pop (neighbors, &numAddedNeighbors) ;
}

return self->searchNumComparisons ;
}



vl_size
vl_kdforest_query_with_array (VlKDForest * self,
vl_uint32 * indexes,
vl_size numNeighbors,
vl_size numQueries,
void * distances,
void const * queries)
{
vl_size numComparisons = 0;
vl_type dataType = vl_kdforest_get_data_type(self) ;
vl_size dimension = vl_kdforest_get_data_dimension(self) ;

#ifdef _OPENMP
#pragma omp parallel default(shared) num_threads(vl_get_max_threads())
#endif
{
vl_index qi ;
vl_size thisNumComparisons = 0 ;
VlKDForestSearcher * searcher ;
VlKDForestNeighbor * neighbors ;

#ifdef _OPENMP
#pragma omp critical
#endif
{
searcher = vl_kdforest_new_searcher(self) ;
neighbors = vl_calloc (sizeof(VlKDForestNeighbor), numNeighbors) ;
}

#ifdef _OPENMP
#pragma omp for
#endif
for(qi = 0 ; qi < (signed)numQueries; ++ qi) {
switch (dataType) {
case VL_TYPE_FLOAT: {
vl_size ni;
thisNumComparisons += vl_kdforestsearcher_query (searcher, neighbors, numNeighbors,
(float const *) (queries) + qi * dimension) ;
for (ni = 0 ; ni < numNeighbors ; ++ni) {
indexes [qi*numNeighbors + ni] = (vl_uint32) neighbors[ni].index ;
if (distances){
*((float*)distances + qi*numNeighbors + ni) = neighbors[ni].distance ;
}
}
break ;
}
case VL_TYPE_DOUBLE: {
vl_size ni;
thisNumComparisons += vl_kdforestsearcher_query (searcher, neighbors, numNeighbors,
(double const *) (queries) + qi * dimension) ;
for (ni = 0 ; ni < numNeighbors ; ++ni) {
indexes [qi*numNeighbors + ni] = (vl_uint32) neighbors[ni].index ;
if (distances){
*((double*)distances + qi*numNeighbors + ni) = neighbors[ni].distance ;
}
}
break ;
}
default:
abort() ;
}
}

#ifdef _OPENMP
#pragma omp critical
#endif
{
numComparisons += thisNumComparisons ;
vl_kdforestsearcher_delete (searcher) ;
vl_free (neighbors) ;
}
}
return numComparisons ;
}



vl_size
vl_kdforest_get_num_nodes_of_tree (VlKDForest const * self, vl_uindex treeIndex)
{
assert (treeIndex < self->numTrees) ;
return self->trees[treeIndex]->numUsedNodes ;
}



vl_size
vl_kdforest_get_depth_of_tree (VlKDForest const * self, vl_uindex treeIndex)
{
assert (treeIndex < self->numTrees) ;
return self->trees[treeIndex]->depth ;
}



vl_size
vl_kdforest_get_num_trees (VlKDForest const * self)
{
return self->numTrees ;
}



void
vl_kdforest_set_max_num_comparisons (VlKDForest * self, vl_size n)
{
self->searchMaxNumComparisons = n ;
}



vl_size
vl_kdforest_get_max_num_comparisons (VlKDForest * self)
{
return self->searchMaxNumComparisons ;
}



void
vl_kdforest_set_thresholding_method (VlKDForest * self, VlKDTreeThresholdingMethod method)
{
assert(method == VL_KDTREE_MEDIAN || method == VL_KDTREE_MEAN) ;
self->thresholdingMethod = method ;
}



VlKDTreeThresholdingMethod
vl_kdforest_get_thresholding_method (VlKDForest const * self)
{
return self->thresholdingMethod ;
}



vl_size
vl_kdforest_get_data_dimension (VlKDForest const * self)
{
return self->dimension ;
}



vl_type
vl_kdforest_get_data_type (VlKDForest const * self)
{
return self->dataType ;
}



VlKDForest *
vl_kdforestsearcher_get_forest (VlKDForestSearcher const * self)
{
return self->forest;
}
