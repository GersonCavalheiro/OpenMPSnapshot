
#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <unordered_map>
#include <vector>

#include "benchmark.h"
#include "bitmap.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"
#include "timer.h"





using namespace std;


void Link(NodeID u, NodeID v, pvector<NodeID>& comp) {
NodeID p1 = comp[u];
NodeID p2 = comp[v];
while (p1 != p2) {
NodeID high = p1 > p2 ? p1 : p2;
NodeID low = p1 + (p2 - high);
NodeID p_high = comp[high];
if ((p_high == low) ||
(p_high == high && compare_and_swap(comp[high], high, low)))
break;
p1 = comp[comp[high]];
p2 = comp[low];
}
}


void Compress(const Graph &g, pvector<NodeID>& comp) {
#pragma omp parallel for schedule(dynamic, 16384)
for (NodeID n = 0; n < g.num_nodes(); n++) {
while (comp[n] != comp[comp[n]]) {
comp[n] = comp[comp[n]];
}
}
}


NodeID SampleFrequentElement(const pvector<NodeID>& comp,
int64_t num_samples = 1024) {
std::unordered_map<NodeID, int> sample_counts(32);
using kvp_type = std::unordered_map<NodeID, int>::value_type;
std::mt19937 gen;
std::uniform_int_distribution<NodeID> distribution(0, comp.size() - 1);
for (NodeID i = 0; i < num_samples; i++) {
NodeID n = distribution(gen);
sample_counts[comp[n]]++;
}
auto most_frequent = std::max_element(
sample_counts.begin(), sample_counts.end(),
[](const kvp_type& a, const kvp_type& b) { return a.second < b.second; });
float frac_of_graph = static_cast<float>(most_frequent->second) / num_samples;
std::cout
<< "Skipping largest intermediate component (ID: " << most_frequent->first
<< ", approx. " << static_cast<int>(frac_of_graph * 100)
<< "% of the graph)" << std::endl;
return most_frequent->first;
}


pvector<NodeID> Afforest(const Graph &g, int32_t neighbor_rounds = 2) {
pvector<NodeID> comp(g.num_nodes());

#pragma omp parallel for
for (NodeID n = 0; n < g.num_nodes(); n++)
comp[n] = n;

for (int r = 0; r < neighbor_rounds; ++r) {
#pragma omp parallel for schedule(dynamic,16384)
for (NodeID u = 0; u < g.num_nodes(); u++) {
for (NodeID v : g.out_neigh(u, r)) {
Link(u, v, comp);
break;
}
}
Compress(g, comp);
}

NodeID c = SampleFrequentElement(comp);

if (!g.directed()) {
#pragma omp parallel for schedule(dynamic, 16384)
for (NodeID u = 0; u < g.num_nodes(); u++) {
if (comp[u] == c)
continue;
for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
Link(u, v, comp);
}
}
} else {
#pragma omp parallel for schedule(dynamic, 16384)
for (NodeID u = 0; u < g.num_nodes(); u++) {
if (comp[u] == c)
continue;
for (NodeID v : g.out_neigh(u, neighbor_rounds)) {
Link(u, v, comp);
}
for (NodeID v : g.in_neigh(u)) {
Link(u, v, comp);
}
}
}
Compress(g, comp);
return comp;
}


void PrintCompStats(const Graph &g, const pvector<NodeID> &comp) {
cout << endl;
unordered_map<NodeID, NodeID> count;
for (NodeID comp_i : comp)
count[comp_i] += 1;
int k = 5;
vector<pair<NodeID, NodeID>> count_vector;
count_vector.reserve(count.size());
for (auto kvp : count)
count_vector.push_back(kvp);
vector<pair<NodeID, NodeID>> top_k = TopK(count_vector, k);
k = min(k, static_cast<int>(top_k.size()));
cout << k << " biggest clusters" << endl;
for (auto kvp : top_k)
cout << kvp.second << ":" << kvp.first << endl;
cout << "There are " << count.size() << " components" << endl;
}


bool CCVerifier(const Graph &g, const pvector<NodeID> &comp) {
unordered_map<NodeID, NodeID> label_to_source;
for (NodeID n : g.vertices())
label_to_source[comp[n]] = n;
Bitmap visited(g.num_nodes());
visited.reset();
vector<NodeID> frontier;
frontier.reserve(g.num_nodes());
for (auto label_source_pair : label_to_source) {
NodeID curr_label = label_source_pair.first;
NodeID source = label_source_pair.second;
frontier.clear();
frontier.push_back(source);
visited.set_bit(source);
for (auto it = frontier.begin(); it != frontier.end(); it++) {
NodeID u = *it;
for (NodeID v : g.out_neigh(u)) {
if (comp[v] != curr_label)
return false;
if (!visited.get_bit(v)) {
visited.set_bit(v);
frontier.push_back(v);
}
}
if (g.directed()) {
for (NodeID v : g.in_neigh(u)) {
if (comp[v] != curr_label)
return false;
if (!visited.get_bit(v)) {
visited.set_bit(v);
frontier.push_back(v);
}
}
}
}
}
for (NodeID n=0; n < g.num_nodes(); n++)
if (!visited.get_bit(n))
return false;
return true;
}


int main(int argc, char* argv[]) {
CLApp cli(argc, argv, "connected-components-afforest");
if (!cli.ParseArgs())
return -1;
Builder b(cli);
Graph g = b.MakeGraph();
auto CCBound = [](const Graph& gr){ return Afforest(gr); };
BenchmarkKernel(cli, g, CCBound, PrintCompStats, CCVerifier);
return 0;
}
