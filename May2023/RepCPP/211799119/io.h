#pragma once
#include <vector>
#include <utility>

#include "lemontc/config.h"

namespace lemon {
void construct_graph_from_arclist(Graph* G,
const std::vector<std::pair<int, int> >& arcs, int node_size);
std::pair<int, int> read_binfile_to_arclist(const char* file_name,
std::vector<std::pair<int, int>>* arcs);
}
