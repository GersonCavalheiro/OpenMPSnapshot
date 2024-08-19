
#ifdef _OPENMP
#define _GLIBCXX_PARALLEL
#endif

#include <algorithm>
#include <cinttypes>
#include <iostream>
#include <vector>

#include "benchmark.h"
#include "builder.h"
#include "command_line.h"
#include "graph.h"
#include "pvector.h"





using namespace std;

size_t OrderedCount(const Graph &g) {
size_t total = 0;
#pragma omp parallel for reduction(+ : total) schedule(dynamic, 64)
for (NodeID u=0; u < g.num_nodes(); u++) {
for (NodeID v : g.out_neigh(u)) {
if (v > u)
break;
auto it = g.out_neigh(u).begin();
for (NodeID w : g.out_neigh(v)) {
if (w > v)
break;
while (*it < w)
it++;
if (w == *it)
total++;
}
}
}
return total;
}


bool WorthRelabelling(const Graph &g) {
int64_t average_degree = g.num_edges() / g.num_nodes();
if (average_degree < 10)
return false;
SourcePicker<Graph> sp(g);
int64_t num_samples = min(int64_t(1000), g.num_nodes());
int64_t sample_total = 0;
pvector<int64_t> samples(num_samples);
for (int64_t trial=0; trial < num_samples; trial++) {
samples[trial] = g.out_degree(sp.PickNext());
sample_total += samples[trial];
}
sort(samples.begin(), samples.end());
double sample_average = static_cast<double>(sample_total) / num_samples;
double sample_median = samples[num_samples/2];
return sample_average / 1.3 > sample_median;
}


size_t Hybrid(const Graph &g) {
if (WorthRelabelling(g))
return OrderedCount(Builder::RelabelByDegree(g));
else
return OrderedCount(g);
}


void PrintTriangleStats(const Graph &g, size_t total_triangles) {
cout << total_triangles << " triangles" << endl;
}


bool TCVerifier(const Graph &g, size_t test_total) {
size_t total = 0;
vector<NodeID> intersection;
intersection.reserve(g.num_nodes());
for (NodeID u : g.vertices()) {
for (NodeID v : g.out_neigh(u)) {
auto new_end = set_intersection(g.out_neigh(u).begin(),
g.out_neigh(u).end(),
g.out_neigh(v).begin(),
g.out_neigh(v).end(),
intersection.begin());
intersection.resize(new_end - intersection.begin());
total += intersection.size();
}
}
total = total / 6;  
if (total != test_total)
cout << total << " != " << test_total << endl;
return total == test_total;
}


int main(int argc, char* argv[]) {
CLApp cli(argc, argv, "triangle count");
if (!cli.ParseArgs())
return -1;
Builder b(cli);
Graph g = b.MakeGraph();
if (g.directed()) {
cout << "Input graph is directed but tc requires undirected" << endl;
return -2;
}
BenchmarkKernel(cli, g, Hybrid, PrintTriangleStats, TCVerifier);
return 0;
}
