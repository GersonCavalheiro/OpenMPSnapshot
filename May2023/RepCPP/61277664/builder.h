
#ifndef BUILDER_H_
#define BUILDER_H_

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <functional>
#include <type_traits>
#include <utility>

#include "timer.h"
#include "command_line.h"
#include "generator.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "reader.h"
#include "mgraph.h"



template <typename VertexID_, typename DestID_ = VertexID_, typename WeightT_ = VertexID_, bool invert = true>
class BuilderBase {
typedef EdgePair<VertexID_, DestID_> Edge;
typedef pvector<Edge> EdgeList;
const CLBase &cli_;
bool symmetrize_;
bool needs_weights_;
int64_t num_vertices_ = -1;

public:
explicit BuilderBase(const CLBase &cli) : cli_(cli) {
symmetrize_ = cli_.symmetrize();
if (symmetrize_) printf("Building a symmetrized graph\n");
needs_weights_ = !std::is_same<VertexID_, DestID_>::value;
}
DestID_ GetSource(EdgePair<VertexID_, VertexID_> e) {
return e.u;
}
DestID_ GetSource(EdgePair<VertexID_, VertexWeight<VertexID_, WeightT_>> e) {
return VertexWeight<VertexID_, WeightT_>(e.u, e.v.w);
}
VertexID_ FindMaxVertexID(const EdgeList &el) {
VertexID_ max_seen = 0;
#pragma omp parallel for reduction(max : max_seen)
for (auto it = el.begin(); it < el.end(); it++) {
Edge e = *it;
max_seen = std::max(max_seen, e.u);
max_seen = std::max(max_seen, (VertexID_) e.v);
}
return max_seen;
}
pvector<VertexID_> CountDegrees(const EdgeList &el, bool transpose) {
pvector<VertexID_> degrees(num_vertices_, 0);
#pragma omp parallel for
for (auto it = el.begin(); it < el.end(); it++) {
Edge e = *it;
if (symmetrize_ || (!symmetrize_ && !transpose))
fetch_and_add(degrees[e.u], 1);
if (symmetrize_ || (!symmetrize_ && transpose))
fetch_and_add(degrees[(VertexID_) e.v], 1);
}
return degrees;
}
static pvector<SGOffset> PrefixSum(const pvector<VertexID_> &degrees) {
pvector<SGOffset> sums(degrees.size() + 1);
SGOffset total = 0;
for (size_t n=0; n < degrees.size(); n++) {
sums[n] = total;
total += degrees[n];
}
sums[degrees.size()] = total;
return sums;
}
static pvector<SGOffset> ParallelPrefixSum(const pvector<VertexID_> &degrees) {
const size_t block_size = 1<<20;
const size_t num_blocks = (degrees.size() + block_size - 1) / block_size;
pvector<SGOffset> local_sums(num_blocks);
#pragma omp parallel for
for (size_t block=0; block < num_blocks; block++) {
SGOffset lsum = 0;
size_t block_end = std::min((block + 1) * block_size, degrees.size());
for (size_t i=block * block_size; i < block_end; i++)
lsum += degrees[i];
local_sums[block] = lsum;
}
pvector<SGOffset> bulk_prefix(num_blocks+1);
SGOffset total = 0;
for (size_t block=0; block < num_blocks; block++) {
bulk_prefix[block] = total;
total += local_sums[block];
}
bulk_prefix[num_blocks] = total;
pvector<SGOffset> prefix(degrees.size() + 1);
#pragma omp parallel for
for (size_t block=0; block < num_blocks; block++) {
SGOffset local_total = bulk_prefix[block];
size_t block_end = std::min((block + 1) * block_size, degrees.size());
for (size_t i=block * block_size; i < block_end; i++) {
prefix[i] = local_total;
local_total += degrees[i];
}
}
prefix[degrees.size()] = bulk_prefix[num_blocks];
return prefix;
}
void OldSquishCSR(const CSRGraph<VertexID_, DestID_, invert> &g, bool transpose, DestID_*** sq_index, DestID_** sq_neighs) {
pvector<VertexID_> diffs(g.num_vertices());
DestID_ *n_start, *n_end;
#pragma omp parallel for private(n_start, n_end)
for (VertexID_ n=0; n < g.num_vertices(); n++) {
if (transpose) {
n_start = g.in_neigh(n).begin();
n_end = g.in_neigh(n).end();
} else {
n_start = g.out_neigh(n).begin();
n_end = g.out_neigh(n).end();
}
std::sort(n_start, n_end);
DestID_ *new_end = std::unique(n_start, n_end);
new_end = std::remove(n_start, new_end, n);
diffs[n] = new_end - n_start;
}
pvector<SGOffset> sq_offsets = ParallelPrefixSum(diffs);
*sq_neighs = new DestID_[sq_offsets[g.num_vertices()]];
*sq_index = GenIndex<VertexID_, DestID_>(sq_offsets, *sq_neighs);
/
void OldMakeCSR(const EdgeList &el, bool transpose, DestID_*** index, DestID_** neighs) {
pvector<VertexID_> degrees = CountDegrees(el, transpose);
pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
*neighs = new DestID_[offsets[num_vertices_]];
*index = GenIndex<VertexID_, DestID_>(offsets, *neighs);
#pragma omp parallel for
for (auto it = el.begin(); it < el.end(); it++) {
Edge e = *it;
if (symmetrize_ || (!symmetrize_ && !transpose))
(*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
if (symmetrize_ || (!symmetrize_ && transpose))
(*neighs)[fetch_and_add(offsets[static_cast<VertexID_>(e.v)], 1)] = GetSource(e);
}
}

void MakeCSR(const EdgeList &el, bool transpose, int** rowptr, DestID_*** index, DestID_** neighs) {
pvector<VertexID_> degrees = CountDegrees(el, transpose);
pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
*neighs = new DestID_[offsets[num_vertices_]];
*index = GenIndex<VertexID_, DestID_>(offsets, *neighs);
*rowptr = new int[num_vertices_+1]; 
for (int i = 0; i < num_vertices_+1; i ++) (*rowptr)[i] = offsets[i];
#pragma omp parallel for
for (auto it = el.begin(); it < el.end(); it++) {
Edge e = *it;
if (symmetrize_ || (!symmetrize_ && !transpose))
(*neighs)[fetch_and_add(offsets[e.u], 1)] = e.v;
if (symmetrize_ || (!symmetrize_ && transpose))
(*neighs)[fetch_and_add(offsets[static_cast<VertexID_>(e.v)], 1)] = GetSource(e);
}
}

void MakeGraphFromEL(EdgeList &el, CSRGraph<VertexID_, DestID_, invert> &g, bool use_dag = false) {
int *rowptr = nullptr, *inv_rowptr = nullptr;
DestID_ **index = nullptr, **inv_index = nullptr;
DestID_ *neighs = nullptr, *inv_neighs = nullptr;
Timer t;
t.Start();
if (num_vertices_ == -1)
num_vertices_ = FindMaxVertexID(el)+1;
if (needs_weights_)
Generator<VertexID_, DestID_, WeightT_>::InsertWeights(el);
MakeCSR(el, false, &rowptr, &index, &neighs);
if (!symmetrize_ && invert)
MakeCSR(el, true, &inv_rowptr, &inv_index, &inv_neighs);
t.Stop();
PrintTime("Build Time", t.Seconds());
if (symmetrize_)
g.Setup(num_vertices_, rowptr, index, neighs);
else
g.Setup(num_vertices_, rowptr, index, neighs, inv_rowptr, inv_index, inv_neighs);
}
void MakeGraph(CSRGraph<VertexID_, DestID_, invert>& new_g, bool use_dag = false) {
CSRGraph<VertexID_, DestID_, invert> g;
{  
EdgeList el;
if (cli_.filename() != "") {
Reader<VertexID_, DestID_, WeightT_, invert> r(cli_.filename());
if ((r.GetSuffix() == ".sg") || (r.GetSuffix() == ".wsg"))
r.ReadSerializedGraph(g);
else el = r.ReadFile(needs_weights_);
} else if (cli_.scale() != -1) {
Generator<VertexID_, DestID_> gen(cli_.scale(), cli_.degree());
el = gen.GenerateEL(cli_.uniform());
}
MakeGraphFromEL(el, g, use_dag);
}
SquishGraph(new_g, g);
}
static
CSRGraph<VertexID_, DestID_, invert> RelabelByDegree(
const CSRGraph<VertexID_, DestID_, invert> &g) {
if (g.directed()) {
std::cout << "Cannot relabel directed graph" << std::endl;
std::exit(-11);
}
Timer t;
t.Start();
typedef std::pair<int64_t, VertexID_> degree_node_p;
pvector<degree_node_p> degree_id_pairs(g.num_vertices());
#pragma omp parallel for
for (VertexID_ n=0; n < g.num_vertices(); n++)
degree_id_pairs[n] = std::make_pair(g.out_degree(n), n);
std::sort(degree_id_pairs.begin(), degree_id_pairs.end(),
std::greater<degree_node_p>());
pvector<VertexID_> degrees(g.num_vertices());
pvector<VertexID_> new_ids(g.num_vertices());
#pragma omp parallel for
for (VertexID_ n=0; n < g.num_vertices(); n++) {
degrees[n] = degree_id_pairs[n].first;
new_ids[degree_id_pairs[n].second] = n;
}
pvector<SGOffset> offsets = ParallelPrefixSum(degrees);
DestID_* neighs = new DestID_[offsets[g.num_vertices()]];
DestID_** index = GenIndex<VertexID_, DestID_>(offsets, neighs);
#pragma omp parallel for
for (VertexID_ u=0; u < g.num_vertices(); u++) {
for (VertexID_ v : g.out_neigh(u))
neighs[offsets[new_ids[u]]++] = new_ids[v];
std::sort(index[new_ids[u]], index[new_ids[u]+1]);
}
t.Stop();
PrintTime("Relabel", t.Seconds());
return CSRGraph<VertexID_, DestID_, invert>(g.num_vertices(), index, neighs);
}
};

typedef BuilderBase<VertexID, VertexID, WeightT> Builder;
typedef BuilderBase<VertexID, WVertex, WeightT> WeightedBuilder;
#endif  
