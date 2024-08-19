#pragma once
#ifndef GRAPH_HPP
#define GRAPH_HPP

#include <iostream>
#include <algorithm>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <climits>
#include <array>
#include <unordered_map>
#include <cstddef>

#include <omp.h>
#include <cstdlib>
#include <unistd.h>

#include "utils.hpp"

class Graph
{
public:
Graph(): edge_indices_(nullptr), edge_list_(nullptr), nv_(-1), ne_(-1)
{}

Graph(GraphElem nv): edge_list_(nullptr), nv_(-1), ne_(-1)
{
edge_indices_   = new GraphElem[nv_+1];
#ifdef USE_OMP_OFFLOAD
#pragma omp target enter data map(to:this[:1])
#pragma omp target enter data map(alloc:edge_indices_[0:nv_+1])
#endif
}

Graph(GraphElem nv, GraphElem ne): 
nv_(nv), ne_(ne) 
{
edge_indices_   = new GraphElem[nv_+1];
edge_list_      = new Edge[ne_];
#ifdef USE_OMP_OFFLOAD
#pragma omp target enter data map(to:this[:1])
#pragma omp target enter data map(alloc:edge_indices_[0:nv_+1])
#pragma omp target enter data map(alloc:edge_list_[0:ne_])
#endif
}

~Graph() 
{

#ifdef USE_OMP_OFFLOAD
#pragma omp target exit data map(delete:edge_indices_[0:nv_+1])
#pragma omp target exit data map(delete:edge_list_[0:ne_])
#endif
delete [] edge_indices_;
delete [] edge_list_;
}

Graph(const Graph &other) = delete;
Graph& operator=(const Graph& d) = delete;

void set_edge_index(GraphElem const vertex, GraphElem const e0)
{
#if defined(DEBUG_BUILD)
assert((vertex >= 0) && (vertex <= nv_));
assert((e0 >= 0) && (e0 <= ne_));
edge_indices_.at(vertex) = e0;
#else
edge_indices_[vertex] = e0;
#endif
} 

void edge_range(GraphElem const vertex, GraphElem& e0, 
GraphElem& e1) const
{
e0 = edge_indices_[vertex];
e1 = edge_indices_[vertex+1];
} 

void set_nedges(GraphElem ne) 
{ 
ne_ = ne;
edge_list_      = new Edge[ne_];
#ifdef USE_OMP_OFFLOAD
#pragma omp target enter data map(alloc:edge_list_[0:ne_])
#endif
}

GraphElem get_nv() const { return nv_; }
GraphElem get_ne() const { return ne_; }


Edge const& get_edge(GraphElem const index) const
{ return edge_list_[index]; }

Edge& set_edge(GraphElem const index)
{ return edge_list_[index]; }       

void print(bool print_weight = true) const
{
if (ne_ < MAX_PRINT_NEDGE)
{
for (GraphElem i = 0; i < nv_; i++)
{
GraphElem e0, e1;
edge_range(i, e0, e1);
if (print_weight) { 
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = get_edge(e);
std::cout << i << " " << edge.tail_ << " " << edge.weight_ << std::endl;
}
}
else { 
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = get_edge(e);
std::cout << i << " " << edge.tail_ << std::endl;
}
}
}
}
else
{
std::cout << "Graph size is {" << nv_ << ", " << ne_ << 
"}, which will overwhelm STDOUT." << std::endl;
}
}

void print_preview() const
{
std::cout << "Printing vertex#0 and all associated edges." << std::endl;
GraphElem e0, e1;
for (GraphElem i = 0; i < nv_; i++)
{
edge_range(i, e0, e1);
if ((e1 - e0) > 0)
{
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = get_edge(e);
std::cout << 0 << " " << edge.tail_ << " " << edge.weight_ << std::endl;
}
break;
}
}
}

void print_stats()
{
std::vector<GraphElem> pdeg(nv_, 0);
for (GraphElem v = 0; v < nv_; v++)
{
GraphElem e0, e1;
edge_range(v, e0, e1);
for (GraphElem e = e0; e < e1; e++)
pdeg[v] += 1;
}

std::sort(pdeg.begin(), pdeg.end());
GraphWeight loc = (GraphWeight)(nv_ + 1)/2.0;
GraphElem median;
if (fmod(loc, 1) != 0)
median = pdeg[(GraphElem)loc]; 
else
median = (pdeg[(GraphElem)floor(loc)] + pdeg[((GraphElem)floor(loc)+1)]) / 2;
GraphElem spdeg = std::accumulate(pdeg.begin(), pdeg.end(), 0);
GraphElem mpdeg = *(std::max_element(pdeg.begin(), pdeg.end()));
std::transform(pdeg.cbegin(), pdeg.cend(), pdeg.cbegin(),
pdeg.begin(), std::multiplies<GraphElem>{});

GraphElem psum_sq = std::accumulate(pdeg.begin(), pdeg.end(), 0);

GraphWeight paverage = (GraphWeight) spdeg / nv_;
GraphWeight pavg_sq  = (GraphWeight) psum_sq / nv_;
GraphWeight pvar     = std::abs(pavg_sq - (paverage*paverage));
GraphWeight pstddev  = sqrt(pvar);

std::cout << std::endl;
std::cout << "--------------------------------------" << std::endl;
std::cout << "Graph characteristics" << std::endl;
std::cout << "--------------------------------------" << std::endl;
std::cout << "Number of vertices: " << nv_ << std::endl;
std::cout << "Number of edges: " << ne_ << std::endl;
std::cout << "Maximum number of edges: " << mpdeg << std::endl;
std::cout << "Median number of edges: " << median << std::endl;
std::cout << "Expected value of X^2: " << pavg_sq << std::endl;
std::cout << "Variance: " << pvar << std::endl;
std::cout << "Standard deviation: " << pstddev << std::endl;
std::cout << "--------------------------------------" << std::endl;
}

GraphElem *edge_indices_;
Edge *edge_list_;

GraphElem get_num_vertices() { return nv_;};
GraphElem get_num_edges() {return ne_;};
GraphElem* get_index_ranges() {return edge_indices_;};
void* get_edge_list() {return edge_list_;};

private:
GraphElem nv_, ne_;
};

class BinaryEdgeList
{
public:
BinaryEdgeList() : 
M_(-1), N_(-1)
{}

Graph* read(std::string binfile, bool isUnitEdgeWeight)
{
std::ifstream file;

file.open(binfile.c_str(), std::ios::in | std::ios::binary); 

if (!file.is_open()) 
{
std::cout << " Error opening file! " << std::endl;
std::abort();
}

file.read(reinterpret_cast<char*>(&M_), sizeof(GraphElem));
file.read(reinterpret_cast<char*>(&N_), sizeof(GraphElem));
#ifdef EDGE_AS_VERTEX_PAIR
GraphElem weighted;
file.read(reinterpret_cast<char*>(&weighted), sizeof(GraphElem));
N_ *= 2;
#endif
Graph *g = new Graph(M_, N_);

uint64_t tot_bytes=(M_+1)*sizeof(GraphElem);
ptrdiff_t offset = 2*sizeof(GraphElem);

if (tot_bytes < INT_MAX)
file.read(reinterpret_cast<char*>(&g->edge_indices_[0]), tot_bytes);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*) &g->edge_indices_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
file.read(reinterpret_cast<char*>(&curr_pointer[offset]), chunk_bytes);
transf_bytes += chunk_bytes;
offset += chunk_bytes;
curr_pointer += chunk_bytes;

if ((tot_bytes - transf_bytes) < INT_MAX)
chunk_bytes = tot_bytes - transf_bytes;
} 
}    

N_ = g->edge_indices_[M_] - g->edge_indices_[0];
g->set_nedges(N_);
tot_bytes = N_*(sizeof(Edge));
offset = 2*sizeof(GraphElem) + (M_+1)*sizeof(GraphElem) + g->edge_indices_[0]*(sizeof(Edge));

#if defined(GRAPH_FT_LOAD)
ptrdiff_t currpos = file.tellg();
ptrdiff_t idx = 0;
GraphElem* vidx = (GraphElem*)malloc(M_ * sizeof(GraphElem));

const int num_sockets = (GRAPH_FT_LOAD == 0) ? 1 : GRAPH_FT_LOAD;
printf("Read file from %d sockets\n", num_sockets);
int n_blocks = num_sockets;

GraphElem NV_blk_sz = M_ / n_blocks;
GraphElem tid_blk_sz = omp_get_num_threads() / n_blocks;

#pragma omp parallel
{
for (int b=0; b<n_blocks; b++) 
{

long NV_beg = b * NV_blk_sz;
long NV_end = std::min(M_, ((b+1) * NV_blk_sz) );
int tid_doit = b * tid_blk_sz;

if (omp_get_thread_num() == tid_doit) 
{
for (GraphElem i = NV_beg; i < NV_end ; i++) 
{
vidx[i] = idx;
const GraphElem vcount = g->edge_indices_[i+1] - g->edge_indices_[i];
idx += vcount;
file.seekg(currpos + vidx[i] * sizeof(Edge), std::ios::beg);
file.read(reinterpret_cast<char*>(&g->edge_list_[vidx[i]]), sizeof(Edge) * (vcount));
}
}
}
}
free(vidx);
#else
if (tot_bytes < INT_MAX)
file.read(reinterpret_cast<char*>(&g->edge_list_[0]), tot_bytes);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*)&g->edge_list_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
file.read(reinterpret_cast<char*>(&curr_pointer[offset]), tot_bytes);
transf_bytes += chunk_bytes;
offset += chunk_bytes;
curr_pointer += chunk_bytes;

if ((tot_bytes - transf_bytes) < INT_MAX)
chunk_bytes = (tot_bytes - transf_bytes);
} 
}   
#endif

file.close();

for(GraphElem i=1;  i < M_+1; i++)
g->edge_indices_[i] -= g->edge_indices_[0];   
g->edge_indices_[0] = 0;

return g;
}
private:
GraphElem M_, N_;
};

class GenerateRGG
{
public:
GenerateRGG(GraphElem nv):
nv_(nv), rn_(0)
{
GraphWeight rc = sqrt((GraphWeight)log(nv_)/(GraphWeight)(PI*nv_));
GraphWeight rt = sqrt((GraphWeight)2.0736/(GraphWeight)nv_);
rn_ = (rc + rt)/(GraphWeight)2.0;

assert(((GraphWeight)1.0) > rn_);
}

Graph* generate(bool isLCG, bool unitEdgeWeight = true, GraphWeight randomEdgePercent = 0.0)
{
std::vector<GraphWeight> X, Y;

X.resize(nv_);
Y.resize(nv_);

Graph *g = new Graph(nv_);

double st = omp_get_wtime();

if (!isLCG) {
seed = (unsigned)reseeder(1);

#if defined(PRINT_RANDOM_XY_COORD)
#pragma omp parallel for
for (GraphElem i = 0; i < nv_; i++) {
X[i] = genRandom<GraphWeight>(0.0, 1.0);
Y[i] = genRandom<GraphWeight>(0.0, 1.0);
std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
}
#else
#pragma omp parallel for
for (GraphElem i = 0; i < nv_; i++) {
X[i] = genRandom<GraphWeight>(0.0, 1.0);
Y[i] = genRandom<GraphWeight>(0.0, 1.0);
}
#endif
}
else { 
LCG xr(1, X.data(), nv_); 

xr.generate();

xr.rescale(Y.data(), nv_, 0);

#if defined(PRINT_RANDOM_XY_COORD)
for (GraphElem i = 0; i < nv_; i++) {
std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
}
#endif
}

double et = omp_get_wtime();
double tt = et - st;

std::cout << "Average time to generate " << nv_ 
<< " random numbers using LCG (in s): " 
<< tt << std::endl;

std::vector<EdgeTuple> edgeList;

#if defined(CHECK_NUM_EDGES)
GraphElem numEdges = 0;
#endif
for (GraphElem i = 0; i < nv_; i++) {
for (GraphElem j = i + 1; j < nv_; j++) {
GraphWeight dx = X[i] - X[j];
GraphWeight dy = Y[i] - Y[j];
GraphWeight ed = sqrt(dx*dx + dy*dy);
if (ed <= rn_) {
if (!unitEdgeWeight) {
edgeList.emplace_back(i, j, ed);
edgeList.emplace_back(j, i, ed);
}
else {
edgeList.emplace_back(i, j);
edgeList.emplace_back(j, i);
}
#if defined(CHECK_NUM_EDGES)
numEdges += 2;
#endif

g->edge_indices_[i+1]++;
g->edge_indices_[j+1]++;
}
}
}

if (randomEdgePercent > 0.0) {
const GraphElem pnedges = (edgeList.size()/2);
const GraphElem nrande = ((GraphElem)(randomEdgePercent * (GraphWeight)pnedges)/100);

#if defined(PRINT_EXTRA_NEDGES)
int extraEdges = 0;
#endif

unsigned rande_seed = (unsigned)(time(0)^getpid());
GraphWeight weight = 1.0;

std::default_random_engine re(rande_seed); 
std::uniform_int_distribution<GraphElem> IR, JR; 
std::uniform_real_distribution<GraphWeight> IJW; 

for (GraphElem k = 0; k < nrande; k++) {

const GraphElem i = (GraphElem)IR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (nv_- 1)});
const GraphElem j = (GraphElem)JR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (nv_- 1)});

if (i == j) 
continue;

auto found = std::find_if(edgeList.begin(), edgeList.end(), 
[&](EdgeTuple const& et) 
{ return ((et.ij_[0] == i) && (et.ij_[1] == j)); });

if (found == std::end(edgeList)) { 

if (!unitEdgeWeight) {
GraphWeight dx = X[i] - X[j];
GraphWeight dy = Y[i] - Y[j];
weight = sqrt(dx*dx + dy*dy);
}

#if defined(PRINT_EXTRA_NEDGES)
extraEdges += 2;
#endif
#if defined(CHECK_NUM_EDGES)
numEdges += 2;
#endif                       
edgeList.emplace_back(i, j, weight);
edgeList.emplace_back(j, i, weight);
g->edge_indices_[i+1]++;
g->edge_indices_[j+1]++;
}
}

#if defined(PRINT_EXTRA_NEDGES)
std::cout << "Adding extra " << (extraEdges/2) << " edges while trying to incorporate " 
<< randomEdgePercent << "%" << " extra edges globally." << std::endl;
#endif
} 

std::partial_sum(g->edge_indices_, g->edge_indices_ + (nv_+1), g->edge_indices_);

for(GraphElem i = 1; i < nv_+1; i++)
g->edge_indices_[i] -= g->edge_indices_[0];   
g->edge_indices_[0] = 0;

g->set_edge_index(0, 0);
for (GraphElem i = 0; i < nv_; i++)
g->set_edge_index(i+1, g->edge_indices_[i+1]);

const GraphElem nedges = g->edge_indices_[nv_] - g->edge_indices_[0];
g->set_nedges(nedges);

auto ecmp = [] (EdgeTuple const& e0, EdgeTuple const& e1)
{ return ((e0.ij_[0] < e1.ij_[0]) || ((e0.ij_[0] == e1.ij_[0]) && (e0.ij_[1] < e1.ij_[1]))); };

if (!std::is_sorted(edgeList.begin(), edgeList.end(), ecmp)) {
#if defined(DEBUG_PRINTF)
std::cout << "Edge list is not sorted." << std::endl;
#endif
std::sort(edgeList.begin(), edgeList.end(), ecmp);
}
#if defined(DEBUG_PRINTF)
else
std::cout << "Edge list is sorted!" << std::endl;
#endif

GraphElem ePos = 0;
for (GraphElem i = 0; i < nv_; i++) {
GraphElem e0, e1;

g->edge_range(i, e0, e1);
#if defined(DEBUG_PRINTF)
if ((i % 100000) == 0)
std::cout << "Processing edges for vertex: " << i << ", range(" << e0 << ", " << e1 <<
")" << std::endl;
#endif
for (GraphElem j = e0; j < e1; j++) {
Edge &edge = g->set_edge(j);

assert(ePos == j);
assert(i == edgeList[ePos].ij_[0]);

edge.tail_ = edgeList[ePos].ij_[1];
edge.weight_ = edgeList[ePos].w_;

ePos++;
}
}

#if defined(CHECK_NUM_EDGES)
const GraphElem ne = g->get_ne();
assert(ne == numEdges);
#endif
edgeList.clear();

X.clear();
Y.clear();

return g;
}

GraphWeight get_d() const { return rn_; }
GraphElem get_nv() const { return nv_; }

private:
GraphElem nv_;
GraphWeight rn_;
};
#endif
