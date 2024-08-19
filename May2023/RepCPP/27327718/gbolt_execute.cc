#include <gbolt.h>
#include <database.h>
#include <common.h>
#include <algorithm>

namespace gbolt {

void GBolt::execute() {
Database *database = Database::get_instance();
vector<Graph> graphs;
vector<Graph> prune_graphs;
#ifdef GBOLT_PERFORMANCE
struct timeval time_start, time_end;
double elapsed = 0.0;
CPU_TIMER_START(elapsed, time_start);
#endif
database->construct_graphs(graphs);
nsupport_ = graphs.size() * support_;
find_frequent_nodes_and_edges(graphs);

database->construct_graphs(frequent_vertex_labels_, frequent_edge_labels_, prune_graphs);
#ifdef GBOLT_PERFORMANCE
CPU_TIMER_END(elapsed, time_start, time_end);
LOG_INFO("gbolt construct graph time: %f", elapsed);
CPU_TIMER_START(elapsed, time_start);
#endif

init_instances(prune_graphs);
project(prune_graphs);
#ifdef GBOLT_PERFORMANCE
CPU_TIMER_END(elapsed, time_start, time_end);
LOG_INFO("gbolt mine graph time: %f", elapsed);
#endif
}

void GBolt::init_instances(const vector<Graph> &graphs) {
#ifdef GBOLT_SERIAL
int num_threads = 1;
#else
int num_threads = omp_get_max_threads();
#endif
gbolt_instances_ = new gbolt_instance_t[num_threads];

int max_edges = 0;
int max_vertice = 0;
for (auto i = 0; i < graphs.size(); ++i) {
max_edges = std::max(graphs[i].get_nedges(), max_edges);
max_vertice = std::max(
static_cast<int>(graphs[i].get_p_vertice()->size()), max_vertice);
}

for (auto i = 0; i < num_threads; ++i) {
#ifdef GBOLT_PERFORMANCE
LOG_INFO("gbolt create thread %d", i);
#endif
string output_file_thread = output_file_ + ".t" + std::to_string(i);
gbolt_instances_[i].history = new History(max_edges, max_vertice);
gbolt_instances_[i].output = new Output(output_file_thread);
gbolt_instances_[i].min_graph = new Graph();
gbolt_instances_[i].min_dfs_codes = new DfsCodes();
gbolt_instances_[i].right_most_path = new Path<int>(DEFAULT_PATH_LEN);
gbolt_instances_[i].min_projection = new MinProjection(DEFAULT_PATH_LEN);
}
}

void GBolt::project(const vector<Graph> &graphs) {
ProjectionMap projection_map;

for (auto i = 0; i < graphs.size(); ++i) {
const Graph &graph = graphs[i];

for (auto j = 0; j < graph.size(); ++j) {
const vertex_t *vertex = graph.get_p_vertex(j);
Edges edges;

if (get_forward_init(*vertex, graph, edges)) {
for (auto k = 0; k < edges.size(); ++k) {
const vertex_t *vertex_from = graph.get_p_vertex(edges[k]->from);
const vertex_t *vertex_to = graph.get_p_vertex(edges[k]->to);
dfs_code_t dfs_code(0, 1, vertex_from->label, edges[k]->label, vertex_to->label);
projection_map[dfs_code].emplace_back(graphs[i].get_id(), edges[k], (const prev_dfs_t *)NULL);
}
}
}
}
int prev_graph_id = -1;
#ifdef GBOLT_SERIAL
int prev_thread_id = 1;
#else
int prev_thread_id = omp_get_thread_num();
#endif
DfsCodes dfs_codes;
#ifndef GBOLT_SERIAL
#pragma omp parallel
#pragma omp single nowait
#endif
{
for (auto it = projection_map.begin(); it != projection_map.end(); ++it) {
Projection &projection = it->second;
int nsupport = count_support(projection);
if (nsupport < nsupport_) {
continue;
}
int from_label = (it->first).from_label;
int edge_label = (it->first).edge_label;
int to_label = (it->first).to_label;
#ifdef GBOLT_SERIAL
dfs_codes.emplace_back(&(it->first));
mine_subgraph(graphs, projection, dfs_codes, nsupport, prev_thread_id, prev_graph_id);
dfs_codes.pop_back();
#else
#pragma omp task shared(graphs, projection, prev_thread_id, prev_graph_id) firstprivate(dfs_codes, nsupport)
{
dfs_codes.emplace_back(&(it->first));
mine_subgraph(graphs, projection, dfs_codes, nsupport, prev_thread_id, prev_graph_id);
}
#endif
}
}
#ifndef GBOLT_SERIAL
#pragma omp taskwait
#endif
}

}  
