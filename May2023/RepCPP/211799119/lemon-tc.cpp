#include <cstring>
#include <tuple>
#include <iostream>

#include "lemontc/io.h"
#include "lemontc/counting.h"

using namespace lemon;  

int main(int argc, char** argv) {
if (argc != 3 || strcmp(argv[1], "-f") != 0) {
std::cout << "Usage: lemon-tc -f input.bin" << std::endl;
exit(-1);
}
FILE *file = fopen(argv[2], "r");
if (file) {
fclose(file);
} else {
std::cout << "file " << argv[2] << " does not exist" << std::endl;
exit(-1);
}
Graph G;
#if VERBOSE && OPENMP
std::cout << "OPENMP enabled." << std::endl;
#endif
int num_nodes, num_edges;
const char* method_hint = std::getenv("METHOD");
{
#if VERBOSE && OPENMP
int actual_num_threads = 0;
#pragma omp parallel shared(actual_num_threads)
{
#pragma omp critical
actual_num_threads = actual_num_threads + 1;
}
std::cout << "Num of threads used: " << actual_num_threads << std::endl;
#endif
std::vector<std::pair<int, int>> arcs;
std::tie(num_nodes, num_edges) = read_binfile_to_arclist(argv[2],
&arcs);
construct_graph_from_arclist(&G, arcs, num_nodes);
}
int64_t tc = 0;
if (method_hint == NULL) {
std::vector<int> degree_list;
int max_degree = collect_degree_info(G, &degree_list, num_nodes);
tc = triangle_count_vertex_iteration(G, degree_list, max_degree);
} else if (strcmp(method_hint, "edge_first") == 0) {
#if VERBOSE
std::cout << "using edge_first method" << std::endl;
#endif
tc = triangle_count(G, num_edges);
} else {  
std::vector<int> degree_list;
int max_degree = collect_degree_info(G, &degree_list, num_nodes);
tc = triangle_count_vertex_iteration(G, degree_list, max_degree);
}
std::cout << "There are " << tc <<
" triangles in the input graph." << std::endl;
return 0;
}
