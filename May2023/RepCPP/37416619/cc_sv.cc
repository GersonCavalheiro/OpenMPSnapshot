
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


pvector<NodeID> ShiloachVishkin(const Graph &g) {
pvector<NodeID> comp(g.num_nodes());
#pragma omp parallel for
for (NodeID n=0; n < g.num_nodes(); n++)
comp[n] = n;
bool change = true;
int num_iter = 0;
while (change) {
change = false;
num_iter++;
#pragma omp parallel for
for (NodeID u=0; u < g.num_nodes(); u++) {
for (NodeID v : g.out_neigh(u)) {
NodeID comp_u = comp[u];
NodeID comp_v = comp[v];
if (comp_u == comp_v) continue;
NodeID high_comp = comp_u > comp_v ? comp_u : comp_v;
NodeID low_comp = comp_u + (comp_v - high_comp);
if (high_comp == comp[high_comp]) {
change = true;
comp[high_comp] = low_comp;
}
}
}
#pragma omp parallel for
for (NodeID n=0; n < g.num_nodes(); n++) {
while (comp[n] != comp[comp[n]]) {
comp[n] = comp[comp[n]];
}
}
}
cout << "Shiloach-Vishkin took " << num_iter << " iterations" << endl;
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
CLApp cli(argc, argv, "connected-components");
if (!cli.ParseArgs())
return -1;
Builder b(cli);
Graph g = b.MakeGraph();
BenchmarkKernel(cli, g, ShiloachVishkin, PrintCompStats, CCVerifier);
return 0;
}
