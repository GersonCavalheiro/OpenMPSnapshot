
#include <cinttypes>
#include <limits>
#include <iostream>
#include <queue>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "platform_atomics.h"
#include "pvector.h"
#include "timer.h"





using namespace std;

const WeightT kDistInf = numeric_limits<WeightT>::max()/2;
const size_t kMaxBin = numeric_limits<size_t>::max()/2;
const size_t kBinSizeThreshold = 1000;

inline
void RelaxEdges(const WGraph &g, NodeID u, WeightT delta,
pvector<WeightT> &dist, vector <vector<NodeID>> &local_bins) {
for (WNode wn : g.out_neigh(u)) {
WeightT old_dist = dist[wn.v];
WeightT new_dist = dist[u] + wn.w;
while (new_dist < old_dist) {
if (compare_and_swap(dist[wn.v], old_dist, new_dist)) {
size_t dest_bin = new_dist/delta;
if (dest_bin >= local_bins.size())
local_bins.resize(dest_bin+1);
local_bins[dest_bin].push_back(wn.v);
break;
}
old_dist = dist[wn.v];      
}
}
}

pvector<WeightT> DeltaStep(const WGraph &g, NodeID source, WeightT delta) {
Timer t;
pvector<WeightT> dist(g.num_nodes(), kDistInf);
dist[source] = 0;
pvector<NodeID> frontier(g.num_edges_directed());
size_t shared_indexes[2] = {0, kMaxBin};
size_t frontier_tails[2] = {1, 0};
frontier[0] = source;
t.Start();
#pragma omp parallel
{
vector<vector<NodeID> > local_bins(0);
size_t iter = 0;
while (shared_indexes[iter&1] != kMaxBin) {
size_t &curr_bin_index = shared_indexes[iter&1];
size_t &next_bin_index = shared_indexes[(iter+1)&1];
size_t &curr_frontier_tail = frontier_tails[iter&1];
size_t &next_frontier_tail = frontier_tails[(iter+1)&1];
#pragma omp for nowait schedule(dynamic, 64)
for (size_t i=0; i < curr_frontier_tail; i++) {
NodeID u = frontier[i];
if (dist[u] >= delta * static_cast<WeightT>(curr_bin_index))
RelaxEdges(g, u, delta, dist, local_bins);
}
while (curr_bin_index < local_bins.size() &&
!local_bins[curr_bin_index].empty() &&
local_bins[curr_bin_index].size() < kBinSizeThreshold) {
vector<NodeID> curr_bin_copy = local_bins[curr_bin_index];
local_bins[curr_bin_index].resize(0);
for (NodeID u : curr_bin_copy)
RelaxEdges(g, u, delta, dist, local_bins);
}
for (size_t i=curr_bin_index; i < local_bins.size(); i++) {
if (!local_bins[i].empty()) {
#pragma omp critical
next_bin_index = min(next_bin_index, i);
break;
}
}
#pragma omp barrier
#pragma omp single nowait
{
t.Stop();
PrintStep(curr_bin_index, t.Millisecs(), curr_frontier_tail);
t.Start();
curr_bin_index = kMaxBin;
curr_frontier_tail = 0;
}
if (next_bin_index < local_bins.size()) {
size_t copy_start = fetch_and_add(next_frontier_tail,
local_bins[next_bin_index].size());
copy(local_bins[next_bin_index].begin(),
local_bins[next_bin_index].end(), frontier.data() + copy_start);
local_bins[next_bin_index].resize(0);
}
iter++;
#pragma omp barrier
}
#pragma omp single
cout << "took " << iter << " iterations" << endl;
}
return dist;
}


void PrintSSSPStats(const WGraph &g, const pvector<WeightT> &dist) {
auto NotInf = [](WeightT d) { return d != kDistInf; };
int64_t num_reached = count_if(dist.begin(), dist.end(), NotInf);
cout << "SSSP Tree reaches " << num_reached << " nodes" << endl;
}


bool SSSPVerifier(const WGraph &g, NodeID source,
const pvector<WeightT> &dist_to_test) {
pvector<WeightT> oracle_dist(g.num_nodes(), kDistInf);
oracle_dist[source] = 0;
typedef pair<WeightT, NodeID> WN;
priority_queue<WN, vector<WN>, greater<WN>> mq;
mq.push(make_pair(0, source));
while (!mq.empty()) {
WeightT td = mq.top().first;
NodeID u = mq.top().second;
mq.pop();
if (td == oracle_dist[u]) {
for (WNode wn : g.out_neigh(u)) {
if (td + wn.w < oracle_dist[wn.v]) {
oracle_dist[wn.v] = td + wn.w;
mq.push(make_pair(td + wn.w, wn.v));
}
}
}
}
bool all_ok = true;
for (NodeID n : g.vertices()) {
if (dist_to_test[n] != oracle_dist[n]) {
cout << n << ": " << dist_to_test[n] << " != " << oracle_dist[n] << endl;
all_ok = false;
}
}
return all_ok;
}


int main(int argc, char* argv[]) {
CLDelta<WeightT> cli(argc, argv, "single-source shortest-path");
if (!cli.ParseArgs())
return -1;
WeightedBuilder b(cli);
WGraph g = b.MakeGraph();
SourcePicker<WGraph> sp(g, cli.start_vertex());
auto SSSPBound = [&sp, &cli] (const WGraph &g) {
return DeltaStep(g, sp.PickNext(), cli.delta());
};
SourcePicker<WGraph> vsp(g, cli.start_vertex());
auto VerifierBound = [&vsp] (const WGraph &g, const pvector<WeightT> &dist) {
return SSSPVerifier(g, vsp.PickNext(), dist);
};
BenchmarkKernel(cli, g, SSSPBound, PrintSSSPStats, VerifierBound);
return 0;
}
