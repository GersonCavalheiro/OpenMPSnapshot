#include "lemontc/counting.h"

#include <vector>
#if VERBOSE || TIMECOUNTING
#include <iostream>
#endif
#if TIMECOUNTING
#include <chrono>  
#endif
#if OPENMP
#include <omp.h>
#endif
namespace lemon {
int triangle_count_given_edge(const Graph& G,
const Graph::Arc& e, const ArcLookUp<Graph>& look_up) {
Graph::Node u = G.source(e);
Graph::Node v = G.target(e);
int t_count = 0;

for (Graph::OutArcIt a(G, u); a != INVALID; ++a) {
Graph::Node u_target = G.target(a);
Graph::Arc a2;
if (G.id(v) < G.id(u_target))
a2 = look_up(v, u_target);
else
a2 = look_up(u_target, v);

t_count += (a2 != INVALID);
}
for (Graph::InArcIt a(G, u); a != INVALID; ++a) {
Graph::Node u_target = G.source(a);
Graph::Arc a2 = look_up(u_target, v);
t_count += (a2 != INVALID);
}
return t_count;
}

int64_t triangle_count(const Graph& G, int total_edge) {
ArcLookUp<Graph> look_up(G);
int64_t triangle_sum = 0;
#if VERBOSE
int64_t iteration_cnt = 0;
int report_unit = total_edge / 100 + 1;
#endif
#if TIMECOUNTING
std::chrono::system_clock::time_point start_time =
std::chrono::system_clock::now();
#endif

#pragma omp parallel for reduction(+:triangle_sum)
for (int a = 0; a < total_edge; ++a) {
triangle_sum += triangle_count_given_edge(G, G.arcFromId(a), look_up);
#if VERBOSE && !defined OPENMP
if (iteration_cnt % report_unit == 1)
std::cout << iteration_cnt * 100 / total_edge <<
"% edges processed" << std::endl;
iteration_cnt++;
#endif
}
#if TIMECOUNTING
std::chrono::system_clock::time_point end_time =
std::chrono::system_clock::now();
std::chrono::system_clock::duration dtn =
end_time - start_time;
using std::chrono::duration_cast;
typedef std::chrono::milliseconds milliseconds;
float time_used = duration_cast<milliseconds>(dtn).count()/1000.0;
std::cout << "Time used: " << time_used << "s" << std::endl;
#endif
return triangle_sum / 3;
}

int triangle_count_given_node(const Graph& G,
const Graph::Node& n, const ArcLookUp<Graph>& look_up,
const std::vector<int>& degree_list,
std::vector<int>* extra_node_list) {
int allowed_node_num = 0;
int n_id = G.id(n);
int degree_n = degree_list[n_id];
for (Graph::OutArcIt a(G, n); a != INVALID; ++a) {
int v_id = G.id(G.target(a));
if (degree_list[v_id] > degree_n ||
(degree_list[v_id] == degree_n && v_id > n_id)) {
(*extra_node_list)[allowed_node_num] = v_id;
allowed_node_num++;
}
}
for (Graph::InArcIt a(G, n); a != INVALID; ++a) {
int v_id = G.id(G.source(a));
if (degree_list[v_id] > degree_n ||
(degree_list[v_id] == degree_n && v_id > n_id)) {
(*extra_node_list)[allowed_node_num] = v_id;
allowed_node_num++;
}
}
int t_count = 0;
for (int i = 0; i < allowed_node_num; i++) {
Graph::Node i_node = G.nodeFromId((*extra_node_list)[i]);
for (int j = i+1; j < allowed_node_num; j++) {
Graph::Node j_node = G.nodeFromId((*extra_node_list)[j]);
Graph::Arc a;
if ((*extra_node_list)[i] < (*extra_node_list)[j])
a = look_up(i_node, j_node);
else
a = look_up(j_node, i_node);
t_count += (a != INVALID);
}
}
return t_count;
}

int64_t triangle_count_vertex_iteration(const Graph& G,
const std::vector<int>& degree_list,
int max_degree) {
ArcLookUp<Graph> look_up(G);
int64_t triangle_sum = 0;
#if VERBOSE
int64_t iteration_cnt = 0;
#endif
int num_nodes = degree_list.size();
int num_nodes_one_percent = num_nodes / 100 + 1;
#if TIMECOUNTING
std::chrono::system_clock::time_point start_time =
std::chrono::system_clock::now();
#endif
#pragma omp parallel
{
std::vector<int> extra_node_list;
extra_node_list.resize(max_degree);
#pragma omp for reduction(+:triangle_sum)
for (int n = 0; n < num_nodes; n++) {
triangle_sum += triangle_count_given_node(G,
G.nodeFromId(n), look_up, degree_list, &extra_node_list);
#if VERBOSE && !defined OPENMP
if (iteration_cnt % num_nodes_one_percent == 1)
std::cout << iteration_cnt * 100 / num_nodes <<
"% nodes processed" << std::endl;
iteration_cnt++;
#endif
}
}
#if TIMECOUNTING
std::chrono::system_clock::time_point end_time =
std::chrono::system_clock::now();
std::chrono::system_clock::duration dtn =
end_time - start_time;
using std::chrono::duration_cast;
typedef std::chrono::milliseconds milliseconds;
float time_used = duration_cast<milliseconds>(dtn).count()/1000.0;
std::cout << "Time used: " << time_used << "s" << std::endl;
#endif
return triangle_sum;
}

int collect_degree_info(const Graph& G, std::vector<int>* degree_list,
int node_size) {
degree_list->resize(node_size, 0);
int max_degree = 0;
for (Graph::ArcIt a(G); a!= INVALID; ++a) {
int u = G.id(G.source(a));
int v = G.id(G.target(a));
(*degree_list)[u]++;
(*degree_list)[v]++;
if ((*degree_list)[u] > max_degree)
max_degree = (*degree_list)[u];
else if ((*degree_list)[v] > max_degree)
max_degree = (*degree_list)[v];
}
return max_degree;
}
}  
