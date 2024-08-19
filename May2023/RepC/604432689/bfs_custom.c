#include "common.h"
#include "oned_csr.h"
#include <mpi.h>
#include <stdint.h>
#include <inttypes.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <limits.h>
#include <assert.h>
static oned_csr_graph g;
void make_graph_data_structure(const tuple_graph* const tg) {
convert_graph_to_oned_csr(tg, &g);
}
void free_graph_data_structure(void) {
free_oned_csr_graph(&g);
}
int bfs_writes_depth_map(void) {
return 0;
}
void run_bfs(int64_t root, int64_t* pred) {
}
void get_vertex_distribution_for_pred(size_t count, const int64_t* vertex_p, int* owner_p, size_t* local_p) {
const int64_t* restrict vertex = vertex_p;
int* restrict owner = owner_p;
size_t* restrict local = local_p;
ptrdiff_t i;
#pragma omp parallel for
for (i = 0; i < (ptrdiff_t)count; ++i) {
owner[i] = VERTEX_OWNER(vertex[i]);
local[i] = VERTEX_LOCAL(vertex[i]);
}
}
int64_t vertex_to_global_for_pred(int v_rank, size_t v_local) {
return VERTEX_TO_GLOBAL(v_rank, v_local);
}
size_t get_nlocalverts_for_pred(void) {
return g.nlocalverts;
}
