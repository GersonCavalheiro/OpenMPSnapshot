
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

#if defined(USE_SHARED_MEMORY)
#include <omp.h>
#include <cstdlib>
#else
#include <mpi.h>
#endif

#include "utils.hpp"

unsigned seed;

#ifdef EDGE_AS_VERTEX_PAIR
struct Edge
{
GraphElem head_, tail_;
GraphWeight weight_;

Edge(): head_(-1), tail_(-1), weight_(0.0) {}
};
#else
struct Edge
{
GraphElem tail_;
GraphWeight weight_;

Edge(): tail_(-1), weight_(0.0) {}
};
#endif

#if defined(USE_SHARED_MEMORY) && defined(ZFILL_CACHE_LINES) && defined(__ARM_ARCH) && __ARM_ARCH >= 8
#ifndef CACHE_LINE_SIZE_BYTES
#define CACHE_LINE_SIZE_BYTES   256
#endif

static const int ZFILL_DISTANCE = 100;


static const int ELEMS_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(GraphWeight);


static const int ZFILL_OFFSET = ZFILL_DISTANCE * ELEMS_PER_CACHE_LINE;

static inline void zfill(GraphWeight * a) 
{ asm volatile("dc zva, %0": : "r"(a)); }
#endif

struct EdgeTuple
{
GraphElem ij_[2];
GraphWeight w_;

EdgeTuple(GraphElem i, GraphElem j, GraphWeight w): 
ij_{i, j}, w_(w)
{}
EdgeTuple(GraphElem i, GraphElem j): 
ij_{i, j}, w_(1.0) 
{}
EdgeTuple(): 
ij_{-1, -1}, w_(0.0)
{}
};


#if defined(USE_SHARED_MEMORY)
class Graph
{
public:
Graph(): nv_(-1), ne_(-1),
edge_indices_(nullptr), edge_list_(nullptr),
vertex_degree_(nullptr), edge_weights_(nullptr)
{}

Graph(GraphElem nv): 
nv_(nv), ne_(-1), 
edge_list_(nullptr), edge_weights_(nullptr)
{
edge_indices_   = new GraphElem[nv_+1];
vertex_degree_  = new GraphWeight[nv_];
}

Graph(GraphElem nv, GraphElem ne): 
nv_(nv), ne_(ne) 
{
edge_indices_   = new GraphElem[nv_+1];
edge_list_      = new Edge[ne_];
vertex_degree_  = new GraphWeight[nv_];
edge_weights_   = new GraphWeight[ne_];
}

~Graph() 
{
delete []edge_indices_;
delete []edge_list_;
delete []edge_weights_;
delete []vertex_degree_;
}

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
edge_weights_   = new GraphWeight[ne_];
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

inline void nbrscan() 
{
#ifdef LLNL_CALIPER_ENABLE
CALI_MARK_BEGIN("nbrscan");
CALI_MARK_BEGIN("parallel");
#endif
GraphElem e0, e1;
#ifdef ENABLE_PREFETCH
#ifdef __INTEL_COMPILER
#pragma noprefetch edge_weights_
#pragma prefetch edge_indices_:3
#pragma prefetch edge_list_:3
#endif
#endif
#ifdef USE_OMP_DYNAMIC
#pragma omp parallel for schedule(dynamic)
#elif defined USE_OMP_TASKLOOP_MASTER
#pragma omp parallel
#pragma omp master
#pragma omp taskloop
#elif defined USE_OMP_TASKS_FOR
#pragma omp parallel
#pragma omp for
#else
#pragma omp parallel for
#endif
for (GraphElem i = 0; i < nv_; i++)
{
#ifdef USE_OMP_TASKS_FOR
#pragma omp task
{
#endif
for (GraphElem e = edge_indices_[i]; e < edge_indices_[i+1]; e++)
{
Edge const& edge = edge_list_[e];
edge_weights_[e] = edge.weight_;
}
#ifdef USE_OMP_TASKS_FOR
}
#endif
}
#ifdef LLNL_CALIPER_ENABLE
CALI_MARK_END("parallel");
CALI_MARK_END("nbrscan");
#endif
}

inline void nbrsum() 
{
#ifdef LLNL_CALIPER_ENABLE
CALI_MARK_BEGIN("nbrsum");
CALI_MARK_BEGIN("parallel");
#endif
GraphElem e0, e1;
#ifdef ENABLE_PREFETCH
#ifdef __INTEL_COMPILER
#pragma noprefetch vertex_degree_
#pragma prefetch edge_indices_:3
#pragma prefetch edge_list_:3
#endif
#endif
#ifdef USE_OMP_DYNAMIC
#pragma omp parallel for schedule(dynamic)
#elif defined USE_OMP_TASKLOOP_MASTER
#pragma omp parallel
#pragma omp master
#pragma omp taskloop
#elif defined USE_OMP_TASKS_FOR
#pragma omp parallel
#pragma omp for
#else
#pragma omp parallel for
#endif
for (GraphElem i = 0; i < nv_; i++)
{
#ifdef USE_OMP_TASKS_FOR
#pragma omp task
{
#endif
for (GraphElem e = edge_indices_[i]; e < edge_indices_[i+1]; e++)
{
Edge const& edge = edge_list_[e];
vertex_degree_[i] += edge.weight_;
}
#ifdef USE_OMP_TASKS_FOR
}
#endif
}
#ifdef LLNL_CALIPER_ENABLE
CALI_MARK_END("parallel");
CALI_MARK_END("nbrsum");
#endif
}

inline void nbrmax() 
{
#ifdef LLNL_CALIPER_ENABLE
CALI_MARK_BEGIN("nbrmax");
CALI_MARK_BEGIN("parallel");
#endif
GraphElem e0, e1;
#ifdef ENABLE_PREFETCH
#ifdef __INTEL_COMPILER
#pragma noprefetch vertex_degree_
#pragma prefetch edge_indices_:3
#pragma prefetch edge_list_:3
#endif
#endif
#ifdef USE_OMP_DYNAMIC
#pragma omp parallel for schedule(dynamic)
#elif defined USE_OMP_TASKLOOP_MASTER
#pragma omp parallel
#pragma omp master
#pragma omp taskloop
#elif defined USE_OMP_TASKS_FOR
#pragma omp parallel
#pragma omp for
#else
#pragma omp parallel for
#endif
for (GraphElem i = 0; i < nv_; i++)
{
#ifdef USE_OMP_TASKS_FOR
#pragma omp task
{
#endif
GraphWeight wmax = -1.0;
for (GraphElem e = edge_indices_[i]; e < edge_indices_[i+1]; e++)
{
Edge const& edge = edge_list_[e];
if (wmax < edge.weight_)
wmax = edge.weight_;
}
vertex_degree_[i] = wmax;
#ifdef USE_OMP_TASKS_FOR
}
#endif
}
#ifdef LLNL_CALIPER_ENABLE
CALI_MARK_END("parallel");
CALI_MARK_END("nbrmax");
#endif
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
GraphWeight *edge_weights_, *vertex_degree_;
private:
GraphElem nv_, ne_;
};

class BinaryEdgeList
{
public:
BinaryEdgeList() : 
M_(-1), N_(-1)
{}

Graph* read(std::string binfile)
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
file.read(&g->edge_list_[0], tot_bytes);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*)&g->edge_list_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
file.read(&curr_poointer[offset], tot_bytes);
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
std::hash<GraphElem> reh;

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

#else 
class Graph
{
public:
Graph(): 
lnv_(-1), lne_(-1), nv_(-1), 
ne_(-1), comm_(MPI_COMM_WORLD) 
{
MPI_Comm_size(comm_, &size_);
MPI_Comm_rank(comm_, &rank_);
}

Graph(GraphElem lnv, GraphElem lne, 
GraphElem nv, GraphElem ne, 
MPI_Comm comm=MPI_COMM_WORLD): 
lnv_(lnv), lne_(lne), 
nv_(nv), ne_(ne), 
comm_(comm) 
{
MPI_Comm_size(comm_, &size_);
MPI_Comm_rank(comm_, &rank_);

degree_ = new GraphWeight[lnv_];
std::memset(degree_, 0, lnv_*sizeof(GraphWeight));
edge_indices_.resize(lnv_+1, 0);
edge_list_.resize(lne_); 

parts_.resize(size_+1);
parts_[0] = 0;

for (GraphElem i = 1; i < size_+1; i++)
parts_[i] = ((nv_ * i) / size_);  
}

~Graph() 
{
delete []degree_;
edge_list_.clear();
edge_indices_.clear();
parts_.clear();
}

void repart(std::vector<GraphElem> const& parts)
{ memcpy(parts_.data(), parts.data(), sizeof(GraphElem)*(size_+1)); }

void set_edge_index(GraphElem const vertex, GraphElem const e0)
{
#if defined(DEBUG_BUILD)
assert((vertex >= 0) && (vertex <= lnv_));
assert((e0 >= 0) && (e0 <= lne_));
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

void set_nedges(GraphElem lne) 
{ 
lne_ = lne; 
edge_list_.resize(lne_);

ne_ = 0;
MPI_Allreduce(&lne_, &ne_, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
}

GraphElem get_base(const int rank) const
{ return parts_[rank]; }

GraphElem get_bound(const int rank) const
{ return parts_[rank+1]; }

GraphElem get_range(const int rank) const
{ return (parts_[rank+1] - parts_[rank] + 1); }

int get_owner(const GraphElem vertex) const
{
const std::vector<GraphElem>::const_iterator iter = 
std::upper_bound(parts_.begin(), parts_.end(), vertex);

return (iter - parts_.begin() - 1);
}

GraphElem get_lnv() const { return lnv_; }
GraphElem get_lne() const { return lne_; }
GraphElem get_nv() const { return nv_; }
GraphElem get_ne() const { return ne_; }
MPI_Comm get_comm() const { return comm_; }


Edge const& get_edge(GraphElem const index) const
{ return edge_list_[index]; }

Edge& set_edge(GraphElem const index)
{ return edge_list_[index]; }       

GraphElem local_to_global(GraphElem idx)
{ return (idx + get_base(rank_)); }

GraphElem global_to_local(GraphElem idx)
{ return (idx - get_base(rank_)); }

GraphElem local_to_global(GraphElem idx, int rank)
{ return (idx + get_base(rank)); }

GraphElem global_to_local(GraphElem idx, int rank)
{ return (idx - get_base(rank)); }

void print(bool print_weight = true) const
{
if (lne_ < MAX_PRINT_NEDGE)
{
for (int p = 0; p < size_; p++)
{
MPI_Barrier(comm_);
if (p == rank_)
{
std::cout << "###############" << std::endl;
std::cout << "Process #" << p << ": " << std::endl;
std::cout << "###############" << std::endl;
GraphElem base = get_base(p);
for (GraphElem i = 0; i < lnv_; i++)
{
GraphElem e0, e1;
edge_range(i, e0, e1);
if (print_weight) { 
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = get_edge(e);
std::cout << i+base << " " << edge.tail_ << " " << edge.weight_ << std::endl;
}
}
else { 
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = get_edge(e);
std::cout << i+base << " " << edge.tail_ << std::endl;
}
}
}
MPI_Barrier(comm_);
}
}
}
else
{
if (rank_ == 0)
std::cout << "Graph size per process is {" << lnv_ << ", " << lne_ << 
"}, which will overwhelm STDOUT." << std::endl;
}
}

void print_dist_stats()
{
GraphElem sumdeg = 0, maxdeg = 0;

MPI_Reduce(&lne_, &sumdeg, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
MPI_Reduce(&lne_, &maxdeg, 1, MPI_GRAPH_TYPE, MPI_MAX, 0, comm_);

GraphElem my_sq = lne_*lne_;
GraphElem sum_sq = 0;
MPI_Reduce(&my_sq, &sum_sq, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

GraphWeight average  = (GraphWeight) sumdeg / size_;
GraphWeight avg_sq   = (GraphWeight) sum_sq / size_;
GraphWeight var      = std::abs(avg_sq - (average*average));
GraphWeight stddev   = sqrt(var);

MPI_Barrier(comm_);

GraphElem pdeg = 0;
for (GraphElem v = 0; v < lnv_; v++)
{
GraphElem e0, e1;
this->edge_range(v, e0, e1);
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = this->get_edge(e);
const int owner = this->get_owner(edge.tail_); 
if (owner != rank_)
{
if (std::find(targets_.begin(), targets_.end(), owner) 
== targets_.end())
{
pdeg += 1;
targets_.push_back(owner);
}
}
}
}

GraphElem spdeg, mpdeg;
MPI_Reduce(&pdeg, &spdeg, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);
MPI_Reduce(&pdeg, &mpdeg, 1, MPI_GRAPH_TYPE, MPI_MAX, 0, comm_);

GraphElem pmy_sq = pdeg*pdeg;
GraphElem psum_sq = 0;
MPI_Reduce(&pmy_sq, &psum_sq, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

GraphWeight paverage = (GraphWeight) spdeg / size_;
GraphWeight pavg_sq  = (GraphWeight) psum_sq / size_;
GraphWeight pvar     = std::abs(pavg_sq - (paverage*paverage));
GraphWeight pstddev  = sqrt(pvar);

MPI_Barrier(comm_);

if (rank_ == 0)
{
std::cout << std::endl;
std::cout << "-------------------------------------------------------" << std::endl;
std::cout << "Graph edge distribution characteristics" << std::endl;
std::cout << "-------------------------------------------------------" << std::endl;
std::cout << "Number of vertices: " << nv_ << std::endl;
std::cout << "Number of edges: " << ne_ << std::endl;
std::cout << "Maximum number of edges: " << maxdeg << std::endl;
std::cout << "Average number of edges: " << average << std::endl;
std::cout << "Expected value of X^2: " << avg_sq << std::endl;
std::cout << "Variance: " << var << std::endl;
std::cout << "Standard deviation: " << stddev << std::endl;
std::cout << "-------------------------------------------------------" << std::endl;
std::cout << "Process graph characteristics" << std::endl;
std::cout << "-------------------------------------------------------" << std::endl;
std::cout << "Number of vertices: " << size_ << std::endl;
std::cout << "Number of edges: " << spdeg << std::endl;
std::cout << "Maximum number of edges: " << mpdeg << std::endl;
std::cout << "Average number of edges: " << paverage << std::endl;
std::cout << "Expected value of X^2: " << pavg_sq << std::endl;
std::cout << "Variance: " << pvar << std::endl;
std::cout << "Standard deviation: " << pstddev << std::endl;
std::cout << "-------------------------------------------------------" << std::endl;
}
}

void rank_order(std::string const& outfile) const
{
std::vector<GraphElem> nbr_pes;

for (GraphElem v = 0; v < lnv_; v++)
{
GraphElem e0, e1;
this->edge_range(v, e0, e1);
for (GraphElem e = e0; e < e1; e++)
{
Edge const& edge = this->get_edge(e);
const int owner = this->get_owner(edge.tail_); 
if (owner != rank_)
{
if (std::find(nbr_pes.begin(), nbr_pes.end(), owner) 
== nbr_pes.end())
{
nbr_pes.push_back(owner);
}
}
}
}

GraphElem nbr_pes_size = nbr_pes.size(), snbr_pes_size = 0;
MPI_Reduce(&nbr_pes_size, &snbr_pes_size, 1, MPI_GRAPH_TYPE, MPI_SUM, 0, comm_);

std::vector<GraphElem> pe_list, pe_map, pe_list_nodup;
std::vector<int> rcounts, rdispls;

if (rank_ == 0)
{
rcounts.resize(size_, 0);
rdispls.resize(size_, 0);
}

MPI_Gather((int*)&nbr_pes_size, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, comm_);

if (rank_ == 0)
{
pe_list.resize(snbr_pes_size, 0);
pe_map.resize(size_, 0);

int idx = 0;
for (int i = 0; i < size_; i++)
{
rdispls[i] = idx;
idx += rcounts[i];
}
}

MPI_Barrier(comm_);

MPI_Gatherv(nbr_pes.data(), nbr_pes_size, MPI_GRAPH_TYPE, 
pe_list.data(), rcounts.data(), rdispls.data(), 
MPI_GRAPH_TYPE, 0, comm_);

if (rank_ == 0)
{
for (GraphElem x = 0; x < snbr_pes_size; x++)
{
if (pe_map[pe_list[x]] == 0)
{
pe_map[pe_list[x]] = 1;
pe_list_nodup.push_back(pe_list[x]);
}
}

std::ofstream ofile;
ofile.open(outfile.c_str(), std::ofstream::out | std::ofstream::app);

for (int p = 0; p < size_-1; p++)
ofile << pe_list_nodup[p] << ",";
ofile << pe_list_nodup[size_-1]; 

ofile.close();

std::cout << "Rank order file: " << outfile << std::endl;
std::cout << "-------------------------------------------------------" << std::endl;
}

MPI_Barrier(comm_);

pe_list.clear();
pe_list_nodup.clear();
nbr_pes.clear();
pe_map.clear();
rcounts.clear();
rdispls.clear();
}

std::vector<GraphElem> edge_indices_;
std::vector<Edge> edge_list_;
GraphWeight* degree_;
private:
GraphElem lnv_, lne_, nv_, ne_;
std::vector<GraphElem> parts_;       
std::vector<int> targets_;       
MPI_Comm comm_; 
int rank_, size_;
};

class BinaryEdgeList
{
public:
BinaryEdgeList() : 
M_(-1), N_(-1), 
M_local_(-1), N_local_(-1), 
comm_(MPI_COMM_WORLD) 
{}
BinaryEdgeList(MPI_Comm comm) : 
M_(-1), N_(-1), 
M_local_(-1), N_local_(-1), 
comm_(comm) 
{}

#ifndef SSTMAC
Graph* read(int me, int nprocs, int ranks_per_node, std::string file)
{
int file_open_error;
MPI_File fh;
MPI_Status status;

#if !defined(SSTMAC)
MPI_Info info;
MPI_Info_create(&info);
int naggr = (ranks_per_node > 1) ? (nprocs/ranks_per_node) : ranks_per_node;
if (naggr >= nprocs)
naggr = 1;
std::stringstream tmp_str;
tmp_str << naggr;
std::string str = tmp_str.str();
MPI_Info_set(info, "cb_nodes", str.c_str());

file_open_error = MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, info, &fh); 
MPI_Info_free(&info);
#else
file_open_error = MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh); 
#endif

if (file_open_error != MPI_SUCCESS) 
{
std::cout << " Error opening file! " << std::endl;
MPI_Abort(comm_, -99);
}

MPI_File_read_all(fh, &M_, sizeof(GraphElem), MPI_BYTE, &status);
MPI_File_read_all(fh, &N_, sizeof(GraphElem), MPI_BYTE, &status);
M_local_ = ((M_*(me + 1)) / nprocs) - ((M_*me) / nprocs); 

Graph *g = new Graph(M_local_, 0, M_, N_);


uint64_t tot_bytes=(M_local_+1)*sizeof(GraphElem);
MPI_Offset offset = 2*sizeof(GraphElem) + ((M_*me) / nprocs)*sizeof(GraphElem);


if (tot_bytes < INT_MAX)
MPI_File_read_at(fh, offset, &g->edge_indices_[0], tot_bytes, MPI_BYTE, &status);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*) &g->edge_indices_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
transf_bytes += chunk_bytes;
offset += chunk_bytes;
curr_pointer += chunk_bytes;

if ((tot_bytes - transf_bytes) < INT_MAX)
chunk_bytes = tot_bytes - transf_bytes;
} 
}    

N_local_ = g->edge_indices_[M_local_] - g->edge_indices_[0];
g->set_nedges(N_local_);

tot_bytes = N_local_*(sizeof(Edge));
offset = 2*sizeof(GraphElem) + (M_+1)*sizeof(GraphElem) + g->edge_indices_[0]*(sizeof(Edge));

if (tot_bytes < INT_MAX)
MPI_File_read_at(fh, offset, &g->edge_list_[0], tot_bytes, MPI_BYTE, &status);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*)&g->edge_list_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
transf_bytes += chunk_bytes;
offset += chunk_bytes;
curr_pointer += chunk_bytes;

if ((tot_bytes - transf_bytes) < INT_MAX)
chunk_bytes = (tot_bytes - transf_bytes);
} 
}    

MPI_File_close(&fh);

for(GraphElem i=1;  i < M_local_+1; i++)
g->edge_indices_[i] -= g->edge_indices_[0];   
g->edge_indices_[0] = 0;

return g;
}

void find_balanced_num_edges(int nprocs, std::string file, std::vector<GraphElem>& mbins)
{
FILE *fp;
GraphElem nv, ne; 
std::vector<GraphElem> nbins(nprocs,0);

fp = fopen(file.c_str(), "rb");
if (fp == NULL) 
{
std::cout<< " Error opening file! " << std::endl;
return;
}

fread(&nv, sizeof(GraphElem), 1, fp);
fread(&ne, sizeof(GraphElem), 1, fp);

GraphElem nbcap = (ne / nprocs), ecount_idx, past_ecount_idx = 0;
int p = 0;

for (GraphElem m = 0; m < nv; m++)
{
fread(&ecount_idx, sizeof(GraphElem), 1, fp);

if ((nbins[p] < nbcap) || (p == (nprocs - 1)))
nbins[p] += (ecount_idx - past_ecount_idx);

if ((nbins[p] >= nbcap) && (p < (nprocs - 1)))
p++;

mbins[p+1]++;
past_ecount_idx = ecount_idx;
}

fclose(fp);

for (int k = 1; k < nprocs+1; k++)
mbins[k] += mbins[k-1]; 

nbins.clear();
}

Graph* read_balanced(int me, int nprocs, int ranks_per_node, std::string file)
{
int file_open_error;
MPI_File fh;
MPI_Status status;
std::vector<GraphElem> mbins(nprocs+1,0);

if (me == 0)
{
find_balanced_num_edges(nprocs, file, mbins);
std::cout << "Trying to achieve equal edge distribution across processes." << std::endl;
}
MPI_Barrier(comm_);
MPI_Bcast(mbins.data(), nprocs+1, MPI_GRAPH_TYPE, 0, comm_);

#if !defined(SSTMAC)
MPI_Info info;
MPI_Info_create(&info);
int naggr = (ranks_per_node > 1) ? (nprocs/ranks_per_node) : ranks_per_node;
if (naggr >= nprocs)
naggr = 1;
std::stringstream tmp_str;
tmp_str << naggr;
std::string str = tmp_str.str();
MPI_Info_set(info, "cb_nodes", str.c_str());

file_open_error = MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, info, &fh); 
MPI_Info_free(&info);
#else
file_open_error = MPI_File_open(comm_, file.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &fh); 
#endif

if (file_open_error != MPI_SUCCESS) 
{
std::cout << " Error opening file! " << std::endl;
MPI_Abort(comm_, -99);
}

MPI_File_read_all(fh, &M_, sizeof(GraphElem), MPI_BYTE, &status);
MPI_File_read_all(fh, &N_, sizeof(GraphElem), MPI_BYTE, &status);
M_local_ = mbins[me+1] - mbins[me];

Graph *g = new Graph(M_local_, 0, M_, N_);
g->repart(mbins);

uint64_t tot_bytes=(M_local_+1)*sizeof(GraphElem);
MPI_Offset offset = 2*sizeof(GraphElem) + mbins[me]*sizeof(GraphElem);

if (tot_bytes < INT_MAX)
MPI_File_read_at(fh, offset, &g->edge_indices_[0], tot_bytes, MPI_BYTE, &status);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*) &g->edge_indices_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
transf_bytes += chunk_bytes;
offset += chunk_bytes;
curr_pointer += chunk_bytes;

if ((tot_bytes - transf_bytes) < INT_MAX)
chunk_bytes = tot_bytes - transf_bytes;
} 
}    

N_local_ = g->edge_indices_[M_local_] - g->edge_indices_[0];
g->set_nedges(N_local_);

tot_bytes = N_local_*(sizeof(Edge));
offset = 2*sizeof(GraphElem) + (M_+1)*sizeof(GraphElem) + g->edge_indices_[0]*(sizeof(Edge));

if (tot_bytes < INT_MAX)
MPI_File_read_at(fh, offset, &g->edge_list_[0], tot_bytes, MPI_BYTE, &status);
else 
{
int chunk_bytes=INT_MAX;
uint8_t *curr_pointer = (uint8_t*)&g->edge_list_[0];
uint64_t transf_bytes = 0;

while (transf_bytes < tot_bytes)
{
MPI_File_read_at(fh, offset, curr_pointer, chunk_bytes, MPI_BYTE, &status);
transf_bytes += chunk_bytes;
offset += chunk_bytes;
curr_pointer += chunk_bytes;

if ((tot_bytes - transf_bytes) < INT_MAX)
chunk_bytes = (tot_bytes - transf_bytes);
} 
}    

MPI_File_close(&fh);

for(GraphElem i=1;  i < M_local_+1; i++)
g->edge_indices_[i] -= g->edge_indices_[0];   
g->edge_indices_[0] = 0;

mbins.clear();

return g;
}
#endif
private:
GraphElem M_;
GraphElem N_;
GraphElem M_local_;
GraphElem N_local_;
MPI_Comm comm_;
};

class GenerateRGG
{
public:
GenerateRGG(GraphElem nv, MPI_Comm comm = MPI_COMM_WORLD)
{
nv_ = nv;
comm_ = comm;

MPI_Comm_rank(comm_, &rank_);
MPI_Comm_size(comm_, &nprocs_);

up_ = down_ = MPI_PROC_NULL;
if (nprocs_ > 1) {
if (rank_ > 0 && rank_ < (nprocs_ - 1)) {
up_ = rank_ - 1;
down_ = rank_ + 1;
}
if (rank_ == 0)
down_ = 1;
if (rank_ == (nprocs_ - 1))
up_ = rank_ - 1;
}
n_ = nv_ / nprocs_;

if ((nv_ % nprocs_) != 0) {
if (rank_ == 0) {
std::cout << "[ERROR] Number of vertices must be perfectly divisible by number of processes." << std::endl;
std::cout << "Exiting..." << std::endl;
}
MPI_Abort(comm_, -99);
}

if (!is_pwr2(nprocs_)) {
if (rank_ == 0) {
std::cout << "[ERROR] Number of processes must be a power of 2." << std::endl;
std::cout << "Exiting..." << std::endl;
}
MPI_Abort(comm_, -99);
}

GraphWeight rc = sqrt((GraphWeight)log(nv)/(GraphWeight)(PI*nv));
GraphWeight rt = sqrt((GraphWeight)2.0736/(GraphWeight)nv);
rn_ = (rc + rt)/(GraphWeight)2.0;

assert(((GraphWeight)1.0/(GraphWeight)nprocs_) > rn_);

MPI_Barrier(comm_);
}

Graph* generate(bool isLCG, bool unitEdgeWeight = true, GraphWeight randomEdgePercent = 0.0)
{
std::vector<GraphWeight> X, Y, X_up, Y_up, X_down, Y_down;

if (isLCG)
X.resize(2*n_);
else
X.resize(n_);

Y.resize(n_);

if (up_ != MPI_PROC_NULL) {
X_up.resize(n_);
Y_up.resize(n_);
}

if (down_ != MPI_PROC_NULL) {
X_down.resize(n_);
Y_down.resize(n_);
}

Graph *g = new Graph(n_, 0, nv_, nv_);

GraphWeight rec_np = (GraphWeight)(1.0/(GraphWeight)nprocs_);
GraphWeight lo = rank_* rec_np; 
GraphWeight hi = lo + rec_np;
assert(hi > lo);

MPI_Barrier(MPI_COMM_WORLD);
double st = MPI_Wtime();

if (!isLCG) {
seed = (unsigned)reseeder(1);

#if defined(PRINT_RANDOM_XY_COORD)
for (int k = 0; k < nprocs_; k++) {
if (k == rank_) {
std::cout << "Random number generated on Process#" << k << " :" << std::endl;
for (GraphElem i = 0; i < n_; i++) {
X[i] = genRandom<GraphWeight>(0.0, 1.0);
Y[i] = genRandom<GraphWeight>(lo, hi);
std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
}
}
MPI_Barrier(comm_);
}
#else
for (GraphElem i = 0; i < n_; i++) {
X[i] = genRandom<GraphWeight>(0.0, 1.0);
Y[i] = genRandom<GraphWeight>(lo, hi);
}
#endif
}
else { 
LCG xr(1, X.data(), 2*n_, comm_); 

xr.generate();

xr.rescale(Y.data(), n_, lo);

#if defined(PRINT_RANDOM_XY_COORD)
for (int k = 0; k < nprocs_; k++) {
if (k == rank_) {
std::cout << "Random number generated on Process#" << k << " :" << std::endl;
for (GraphElem i = 0; i < n_; i++) {
std::cout << "X, Y: " << X[i] << ", " << Y[i] << std::endl;
}
}
MPI_Barrier(comm_);
}
#endif
}

double et = MPI_Wtime();
double tt = et - st;
double tot_tt = 0.0;
MPI_Reduce(&tt, &tot_tt, 1, MPI_DOUBLE, MPI_SUM, 0, comm_);

if (rank_ == 0) {
double tot_avg = (tot_tt/nprocs_);
std::cout << "Average time to generate " << 2*n_ 
<< " random numbers using LCG (in s): " 
<< tot_avg << std::endl;
}


std::vector<EdgeTuple> sendup_edges, senddn_edges; 
std::vector<EdgeTuple> recvup_edges, recvdn_edges;
std::vector<EdgeTuple> edgeList;

std::array<GraphElem, 2> send_sizes = {0, 0}, recv_sizes = {0, 0};
#if defined(CHECK_NUM_EDGES)
GraphElem numEdges = 0;
#endif
for (GraphElem i = 0; i < n_; i++) {
for (GraphElem j = i + 1; j < n_; j++) {
GraphWeight dx = X[i] - X[j];
GraphWeight dy = Y[i] - Y[j];
GraphWeight ed = sqrt(dx*dx + dy*dy);
if (ed <= rn_) {
const GraphElem g_i = g->local_to_global(i);
const GraphElem g_j = g->local_to_global(j);

if (!unitEdgeWeight) {
edgeList.emplace_back(i, g_j, ed);
edgeList.emplace_back(j, g_i, ed);
}
else {
edgeList.emplace_back(i, g_j);
edgeList.emplace_back(j, g_i);
}
#if defined(CHECK_NUM_EDGES)
numEdges += 2;
#endif

g->edge_indices_[i+1]++;
g->edge_indices_[j+1]++;
}
}
}

MPI_Barrier(comm_);


const int x_ndown   = X_down.empty() ? 0 : n_;
const int y_ndown   = Y_down.empty() ? 0 : n_;
const int x_nup     = X_up.empty() ? 0 : n_;
const int y_nup     = Y_up.empty() ? 0 : n_;

#ifndef DONT_USE_SENDRECV
MPI_Sendrecv(X.data(), n_, MPI_WEIGHT_TYPE, up_, SR_X_UP_TAG, 
X_down.data(), x_ndown, MPI_WEIGHT_TYPE, down_, SR_X_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Sendrecv(X.data(), n_, MPI_WEIGHT_TYPE, down_, SR_X_DOWN_TAG, 
X_up.data(), x_nup, MPI_WEIGHT_TYPE, up_, SR_X_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Sendrecv(Y.data(), n_, MPI_WEIGHT_TYPE, up_, SR_Y_UP_TAG, 
Y_down.data(), y_ndown, MPI_WEIGHT_TYPE, down_, SR_Y_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Sendrecv(Y.data(), n_, MPI_WEIGHT_TYPE, down_, SR_Y_DOWN_TAG, 
Y_up.data(), y_nup, MPI_WEIGHT_TYPE, up_, SR_Y_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
#else
if (rank_% 2 == 0)
{
MPI_Send(X.data(), n_, MPI_WEIGHT_TYPE, up_, SR_X_UP_TAG, comm_);
MPI_Send(X.data(), n_, MPI_WEIGHT_TYPE, down_, SR_X_DOWN_TAG, comm_); 
MPI_Send(Y.data(), n_, MPI_WEIGHT_TYPE, up_, SR_Y_UP_TAG, comm_);
MPI_Send(Y.data(), n_, MPI_WEIGHT_TYPE, down_, SR_Y_DOWN_TAG, comm_);

MPI_Recv(X_down.data(), x_ndown, MPI_WEIGHT_TYPE, down_, SR_X_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(X_up.data(), x_nup, MPI_WEIGHT_TYPE, up_, SR_X_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(Y_down.data(), y_ndown, MPI_WEIGHT_TYPE, down_, SR_Y_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(Y_up.data(), y_nup, MPI_WEIGHT_TYPE, up_, SR_Y_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
}
else
{

MPI_Recv(X_down.data(), x_ndown, MPI_WEIGHT_TYPE, down_, SR_X_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(X_up.data(), x_nup, MPI_WEIGHT_TYPE, up_, SR_X_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(Y_down.data(), y_ndown, MPI_WEIGHT_TYPE, down_, SR_Y_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(Y_up.data(), y_nup, MPI_WEIGHT_TYPE, up_, SR_Y_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);

MPI_Send(X.data(), n_, MPI_WEIGHT_TYPE, up_, SR_X_UP_TAG, comm_);
MPI_Send(X.data(), n_, MPI_WEIGHT_TYPE, down_, SR_X_DOWN_TAG, comm_); 
MPI_Send(Y.data(), n_, MPI_WEIGHT_TYPE, up_, SR_Y_UP_TAG, comm_);
MPI_Send(Y.data(), n_, MPI_WEIGHT_TYPE, down_, SR_Y_DOWN_TAG, comm_);
}
#endif

if (nprocs_ > 1) {
if (up_ != MPI_PROC_NULL) {

for (GraphElem i = 0; i < n_; i++) {
for (GraphElem j = i + 1; j < n_; j++) {
GraphWeight dx = X[i] - X_up[j];
GraphWeight dy = Y[i] - Y_up[j];
GraphWeight ed = sqrt(dx*dx + dy*dy);

if (ed <= rn_) {
const GraphElem g_i = g->local_to_global(i);
const GraphElem g_j = j + up_*n_;

if (!unitEdgeWeight) {
sendup_edges.emplace_back(j, g_i, ed);
edgeList.emplace_back(i, g_j, ed);
}
else {
sendup_edges.emplace_back(j, g_i);
edgeList.emplace_back(i, g_j);
}
#if defined(CHECK_NUM_EDGES)
numEdges++;
#endif
g->edge_indices_[i+1]++;
}
}
}

send_sizes[0] = sendup_edges.size();
}

if (down_ != MPI_PROC_NULL) {

for (GraphElem i = 0; i < n_; i++) {
for (GraphElem j = i + 1; j < n_; j++) {
GraphWeight dx = X[i] - X_down[j];
GraphWeight dy = Y[i] - Y_down[j];
GraphWeight ed = sqrt(dx*dx + dy*dy);

if (ed <= rn_) {
const GraphElem g_i = g->local_to_global(i);
const GraphElem g_j = j + down_*n_;

if (!unitEdgeWeight) {
senddn_edges.emplace_back(j, g_i, ed);
edgeList.emplace_back(i, g_j, ed);
}
else {
senddn_edges.emplace_back(j, g_i);
edgeList.emplace_back(i, g_j);
}
#if defined(CHECK_NUM_EDGES)
numEdges++;
#endif
g->edge_indices_[i+1]++;
}
}
}

send_sizes[1] = senddn_edges.size();
}
}

MPI_Barrier(comm_);

#ifndef SSTMAC 
MPI_Sendrecv(&send_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_UP_TAG, 
&recv_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Sendrecv(&send_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_DOWN_TAG, 
&recv_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
#else
if (rank_ % 2 == 0)
{
MPI_Send(&send_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_UP_TAG, comm_); 
MPI_Send(&send_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_DOWN_TAG, comm_); 
MPI_Recv(&recv_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(&recv_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
}
else
{
MPI_Recv(&recv_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_UP_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Recv(&recv_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_DOWN_TAG, 
comm_, MPI_STATUS_IGNORE);
MPI_Send(&send_sizes[0], 1, MPI_GRAPH_TYPE, up_, SR_SIZES_UP_TAG, comm_); 
MPI_Send(&send_sizes[1], 1, MPI_GRAPH_TYPE, down_, SR_SIZES_DOWN_TAG, comm_); 
}
#endif


if (recv_sizes[0] > 0)
recvup_edges.resize(recv_sizes[0]);
if (recv_sizes[1] > 0)
recvdn_edges.resize(recv_sizes[1]);

MPI_Datatype edgeType;
EdgeTuple einfo;
MPI_Aint begin, s, t, w;
MPI_Get_address(&einfo, &begin);
MPI_Get_address(&einfo.ij_[0], &s);
MPI_Get_address(&einfo.ij_[1], &t);
MPI_Get_address(&einfo.w_, &w);

int blens[] = { 1, 1, 1 };
MPI_Aint displ[] = { s - begin, t - begin, w - begin };
MPI_Datatype types[] = { MPI_GRAPH_TYPE, MPI_GRAPH_TYPE, MPI_WEIGHT_TYPE };

MPI_Type_create_struct(3, blens, displ, types, &edgeType);
MPI_Type_commit(&edgeType);

#ifndef DONT_USE_SENDRECV
MPI_Sendrecv(sendup_edges.data(), send_sizes[0], edgeType, 
up_, SR_UP_TAG, recvdn_edges.data(), recv_sizes[1], 
edgeType, down_, SR_UP_TAG, comm_, MPI_STATUS_IGNORE);
MPI_Sendrecv(senddn_edges.data(), send_sizes[1], edgeType, 
down_, SR_DOWN_TAG, recvup_edges.data(), recv_sizes[0], 
edgeType, up_, SR_DOWN_TAG, comm_, MPI_STATUS_IGNORE);
#else
if (rank_ % 2 == 0)
{
MPI_Send(sendup_edges.data(), send_sizes[0], edgeType, up_, SR_UP_TAG, comm_);
MPI_Send(senddn_edges.data(), send_sizes[1], edgeType, down_, SR_DOWN_TAG, comm_);
MPI_Recv(recvdn_edges.data(), recv_sizes[1], edgeType, down_, SR_UP_TAG, comm_, MPI_STATUS_IGNORE);
MPI_Recv(recvup_edges.data(), recv_sizes[0], edgeType, up_, SR_DOWN_TAG, comm_, MPI_STATUS_IGNORE);
}
else
{
MPI_Recv(recvdn_edges.data(), recv_sizes[1], edgeType, down_, SR_UP_TAG, comm_, MPI_STATUS_IGNORE);
MPI_Recv(recvup_edges.data(), recv_sizes[0], edgeType, up_, SR_DOWN_TAG, comm_, MPI_STATUS_IGNORE);
MPI_Send(sendup_edges.data(), send_sizes[0], edgeType, up_, SR_UP_TAG, comm_);
MPI_Send(senddn_edges.data(), send_sizes[1], edgeType, down_, SR_DOWN_TAG, comm_);
}
#endif

MPI_Type_free(&edgeType);


if (down_ != MPI_PROC_NULL) {
for (GraphElem i = 0; i < recv_sizes[1]; i++) {
#if defined(CHECK_NUM_EDGES)
numEdges++;
#endif           
if (!unitEdgeWeight)
edgeList.emplace_back(recvdn_edges[i].ij_[0], recvdn_edges[i].ij_[1], recvdn_edges[i].w_);
else
edgeList.emplace_back(recvdn_edges[i].ij_[0], recvdn_edges[i].ij_[1]);
g->edge_indices_[recvdn_edges[i].ij_[0]+1]++; 
} 
}

if (up_ != MPI_PROC_NULL) {
for (GraphElem i = 0; i < recv_sizes[0]; i++) {
#if defined(CHECK_NUM_EDGES)
numEdges++;
#endif
if (!unitEdgeWeight)
edgeList.emplace_back(recvup_edges[i].ij_[0], recvup_edges[i].ij_[1], recvup_edges[i].w_);
else
edgeList.emplace_back(recvup_edges[i].ij_[0], recvup_edges[i].ij_[1]);
g->edge_indices_[recvup_edges[i].ij_[0]+1]++; 
}
}

if (randomEdgePercent > 0.0) {
const GraphElem pnedges = (edgeList.size()/2);
GraphElem tot_pnedges = 0;

MPI_Allreduce(&pnedges, &tot_pnedges, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);

const GraphElem nrande = ((GraphElem)(randomEdgePercent * (GraphWeight)tot_pnedges)/100);

GraphElem pnrande = 0;

if (nrande < nprocs_) {
if (rank_ == (nprocs_ - 1))
pnrande += nrande;
}
else {
pnrande = nrande / nprocs_;
const GraphElem pnrem = nrande % nprocs_;
if (pnrem != 0) {
if (rank_ == (nprocs_ - 1))
pnrande += pnrem;
}
}


std::vector<std::vector<EdgeTuple>> rand_edges(nprocs_); 
std::vector<EdgeTuple> sendrand_edges, recvrand_edges;

std::vector<int> sendrand_sizes(nprocs_), recvrand_sizes(nprocs_);

#if defined(PRINT_EXTRA_NEDGES)
int extraEdges = 0;
#endif

#if defined(DEBUG_PRINTF)
for (int i = 0; i < nprocs_; i++) {
if (i == rank_) {
std::cout << "[" << i << "]Target process for random edge insertion between " 
<< lo << " and " << hi << std::endl;
}
MPI_Barrier(comm_);
}
#endif
unsigned rande_seed = (unsigned)(time(0)^getpid());
GraphWeight weight = 1.0;
std::hash<GraphElem> reh;

std::default_random_engine re(rande_seed); 
std::uniform_int_distribution<GraphElem> IR, JR; 
std::uniform_real_distribution<GraphWeight> IJW; 

for (GraphElem k = 0; k < pnrande; k++) {

const GraphElem i = (GraphElem)IR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (n_- 1)});
const GraphElem g_j = (GraphElem)JR(re, std::uniform_int_distribution<GraphElem>::param_type{0, (nv_- 1)});
const int target = g->get_owner(g_j);
const GraphElem j = g->global_to_local(g_j, target); 

if (i == j) 
continue;

const GraphElem g_i = g->local_to_global(i);

auto found = std::find_if(edgeList.begin(), edgeList.end(), 
[&](EdgeTuple const& et) 
{ return ((et.ij_[0] == i) && (et.ij_[1] == g_j)); });

if (found == std::end(edgeList)) { 

if (!unitEdgeWeight) {
if (target == rank_) {
GraphWeight dx = X[i] - X[j];
GraphWeight dy = Y[i] - Y[j];
weight = sqrt(dx*dx + dy*dy);
}
else if (target == up_) {
GraphWeight dx = X[i] - X_up[j];
GraphWeight dy = Y[i] - Y_up[j];
weight = sqrt(dx*dx + dy*dy);
}
else if (target == down_) {
GraphWeight dx = X[i] - X_down[j];
GraphWeight dy = Y[i] - Y_down[j];
weight = sqrt(dx*dx + dy*dy);
}
else {
unsigned randw_seed = reh((GraphElem)(g_i*nv_+g_j));
std::default_random_engine rew(randw_seed); 
weight = (GraphWeight)IJW(rew, std::uniform_real_distribution<GraphWeight>::param_type{0.01, 1.0});
}
}

rand_edges[target].emplace_back(j, g_i, weight);
sendrand_sizes[target]++;

#if defined(PRINT_EXTRA_NEDGES)
extraEdges++;
#endif
#if defined(CHECK_NUM_EDGES)
numEdges++;
#endif                       
edgeList.emplace_back(i, g_j, weight);
g->edge_indices_[i+1]++;
}
}

#if defined(PRINT_EXTRA_NEDGES)
int totExtraEdges = 0;
MPI_Reduce(&extraEdges, &totExtraEdges, 1, MPI_INT, MPI_SUM, 0, comm_);
if (rank_ == 0)
std::cout << "Adding extra " << totExtraEdges << " edges while trying to incorporate " 
<< randomEdgePercent << "%" << " extra edges globally." << std::endl;
#endif

MPI_Barrier(comm_);

MPI_Request rande_sreq;

MPI_Ialltoall(sendrand_sizes.data(), 1, MPI_INT, 
recvrand_sizes.data(), 1, MPI_INT, comm_, 
&rande_sreq);

for (int p = 0; p < nprocs_; p++) {
sendrand_edges.insert(sendrand_edges.end(), 
rand_edges[p].begin(), rand_edges[p].end());
}

MPI_Wait(&rande_sreq, MPI_STATUS_IGNORE);

const int rcount = std::accumulate(recvrand_sizes.begin(), recvrand_sizes.end(), 0);
recvrand_edges.resize(rcount);


int rpos = 0, spos = 0;
std::vector<int> sdispls(nprocs_), rdispls(nprocs_);

for (int p = 0; p < nprocs_; p++) {

sendrand_sizes[p] *= sizeof(struct EdgeTuple);
recvrand_sizes[p] *= sizeof(struct EdgeTuple);

sdispls[p] = spos;
rdispls[p] = rpos;

spos += sendrand_sizes[p];
rpos += recvrand_sizes[p];
}

MPI_Alltoallv(sendrand_edges.data(), sendrand_sizes.data(), sdispls.data(), 
MPI_BYTE, recvrand_edges.data(), recvrand_sizes.data(), rdispls.data(), 
MPI_BYTE, comm_);

for (int i = 0; i < rcount; i++) {
#if defined(CHECK_NUM_EDGES)
numEdges++;
#endif
edgeList.emplace_back(recvrand_edges[i].ij_[0], recvrand_edges[i].ij_[1], recvrand_edges[i].w_);
g->edge_indices_[recvrand_edges[i].ij_[0]+1]++; 
}

sendrand_edges.clear();
recvrand_edges.clear();
rand_edges.clear();
} 

MPI_Barrier(comm_);

std::partial_sum(g->edge_indices_.begin(), g->edge_indices_.end(), g->edge_indices_.begin());

for(GraphElem i = 1; i < n_+1; i++)
g->edge_indices_[i] -= g->edge_indices_[0];   
g->edge_indices_[0] = 0;

g->set_edge_index(0, 0);
for (GraphElem i = 0; i < n_; i++)
g->set_edge_index(i+1, g->edge_indices_[i+1]);

const GraphElem nedges = g->edge_indices_[n_] - g->edge_indices_[0];
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
for (GraphElem i = 0; i < n_; i++) {
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
GraphElem tot_numEdges = 0;
MPI_Allreduce(&numEdges, &tot_numEdges, 1, MPI_GRAPH_TYPE, MPI_SUM, comm_);
const GraphElem tne = g->get_ne();
assert(tne == tot_numEdges);
#endif
edgeList.clear();

X.clear();
Y.clear();
X_up.clear();
Y_up.clear();
X_down.clear();
Y_down.clear();

sendup_edges.clear();
senddn_edges.clear();
recvup_edges.clear();
recvdn_edges.clear();

return g;
}

GraphWeight get_d() const { return rn_; }
GraphElem get_nv() const { return nv_; }

private:
GraphElem nv_, n_;
GraphWeight rn_;
MPI_Comm comm_;
int nprocs_, rank_, up_, down_;
};
#endif
#endif
