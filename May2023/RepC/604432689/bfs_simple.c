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
static int64_t* g_oldq;
static int64_t* g_newq;
static unsigned long* g_visited;
static const int coalescing_size = 256;
static int64_t* g_outgoing;
static size_t* g_outgoing_counts ;
static MPI_Request* g_outgoing_reqs;
static int* g_outgoing_reqs_active;
static int64_t* g_recvbuf;
void make_graph_data_structure(const tuple_graph* const tg) {
convert_graph_to_oned_csr(tg, &g);
const size_t nlocalverts = g.nlocalverts;
g_oldq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
g_newq = (int64_t*)xmalloc(nlocalverts * sizeof(int64_t));
const int ulong_bits = sizeof(unsigned long) * CHAR_BIT;
int64_t visited_size = (nlocalverts + ulong_bits - 1) / ulong_bits;
g_visited = (unsigned long*)xmalloc(visited_size * sizeof(unsigned long));
g_outgoing = (int64_t*)xMPI_Alloc_mem(coalescing_size * size * 2 * sizeof(int64_t));
g_outgoing_counts = (size_t*)xmalloc(size * sizeof(size_t)) ;
g_outgoing_reqs = (MPI_Request*)xmalloc(size * sizeof(MPI_Request));
g_outgoing_reqs_active = (int*)xmalloc(size * sizeof(int));
g_recvbuf = (int64_t*)xMPI_Alloc_mem(coalescing_size * 2 * sizeof(int64_t));
}
void free_graph_data_structure(void) {
free(g_oldq);
free(g_newq);
free(g_visited);
MPI_Free_mem(g_outgoing);
free(g_outgoing_counts);
free(g_outgoing_reqs);
free(g_outgoing_reqs_active);
MPI_Free_mem(g_recvbuf);
free_oned_csr_graph(&g);
}
int bfs_writes_depth_map(void) {
return 0;
}
void run_bfs(int64_t root, int64_t* pred) {
const size_t nlocalverts = g.nlocalverts;
int64_t* restrict oldq = g_oldq;
int64_t* restrict newq = g_newq;
size_t oldq_count = 0;
size_t newq_count = 0;
const int ulong_bits = sizeof(unsigned long) * CHAR_BIT;
int64_t visited_size = (nlocalverts + ulong_bits - 1) / ulong_bits;
unsigned long* restrict visited = g_visited;
memset(visited, 0, visited_size * sizeof(unsigned long));
#define SET_VISITED(v) do {visited[VERTEX_LOCAL((v)) / ulong_bits] |= (1UL << (VERTEX_LOCAL((v)) % ulong_bits));} while (0)
#define TEST_VISITED(v) ((visited[VERTEX_LOCAL((v)) / ulong_bits] & (1UL << (VERTEX_LOCAL((v)) % ulong_bits))) != 0)
const int coalescing_size = 256;
int64_t* restrict outgoing = g_outgoing;
size_t* restrict outgoing_counts = g_outgoing_counts;
MPI_Request* restrict outgoing_reqs = g_outgoing_reqs;
int* restrict outgoing_reqs_active = g_outgoing_reqs_active;
memset(outgoing_reqs_active, 0, size * sizeof(int));
int64_t* restrict recvbuf = g_recvbuf;
MPI_Request recvreq;
int recvreq_active = 0;
int num_ranks_done;
{size_t i; for (i = 0; i < nlocalverts; ++i) pred[i] = -1;}
if (VERTEX_OWNER(root) == rank) {
SET_VISITED(root);
pred[VERTEX_LOCAL(root)] = root;
oldq[oldq_count++] = root;
}
#define CHECK_MPI_REQS \
\
do { \
\
while (recvreq_active) { \
int flag; \
MPI_Status st; \
MPI_Test(&recvreq, &flag, &st); \
if (flag) { \
recvreq_active = 0; \
int count; \
MPI_Get_count(&st, MPI_INT64_T, &count); \
\
if (count == 0) { \
++num_ranks_done; \
} else { \
int j; \
for (j = 0; j < count; j += 2) { \
int64_t tgt = recvbuf[j]; \
int64_t src = recvbuf[j + 1]; \
\
assert (VERTEX_OWNER(tgt) == rank); \
if (!TEST_VISITED(tgt)) { \
SET_VISITED(tgt); \
pred[VERTEX_LOCAL(tgt)] = src; \
newq[newq_count++] = tgt; \
} \
} \
} \
\
if (num_ranks_done < size) { \
MPI_Irecv(recvbuf, coalescing_size * 2, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recvreq); \
recvreq_active = 1; \
} \
} else break; \
} \
\
int c; \
for (c = 0; c < size; ++c) { \
if (outgoing_reqs_active[c]) { \
int flag; \
MPI_Test(&outgoing_reqs[c], &flag, MPI_STATUS_IGNORE); \
if (flag) outgoing_reqs_active[c] = 0; \
} \
} \
} while (0)
while (1) {
memset(outgoing_counts, 0, size * sizeof(size_t));
num_ranks_done = 1; 
if (num_ranks_done < size) {
MPI_Irecv(recvbuf, coalescing_size * 2, MPI_INT64_T, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &recvreq);
recvreq_active = 1;
}
size_t i;
for (i = 0; i < oldq_count; ++i) {
CHECK_MPI_REQS;
assert (VERTEX_OWNER(oldq[i]) == rank);
assert (pred[VERTEX_LOCAL(oldq[i])] >= 0 && pred[VERTEX_LOCAL(oldq[i])] < g.nglobalverts);
int64_t src = oldq[i];
size_t j, j_end = g.rowstarts[VERTEX_LOCAL(oldq[i]) + 1];
for (j = g.rowstarts[VERTEX_LOCAL(oldq[i])]; j < j_end; ++j) {
int64_t tgt = g.column[j];
int owner = VERTEX_OWNER(tgt);
if (owner == rank) {
if (!TEST_VISITED(tgt)) {
SET_VISITED(tgt);
pred[VERTEX_LOCAL(tgt)] = src;
newq[newq_count++] = tgt;
}
} else {
while (outgoing_reqs_active[owner]) CHECK_MPI_REQS; 
size_t c = outgoing_counts[owner];
outgoing[owner * coalescing_size * 2 + c] = tgt;
outgoing[owner * coalescing_size * 2 + c + 1] = src;
outgoing_counts[owner] += 2;
if (outgoing_counts[owner] == coalescing_size * 2) {
MPI_Isend(&outgoing[owner * coalescing_size * 2], coalescing_size * 2, MPI_INT64_T, owner, 0, MPI_COMM_WORLD, &outgoing_reqs[owner]);
outgoing_reqs_active[owner] = 1;
outgoing_counts[owner] = 0;
}
}
}
}
int offset;
for (offset = 1; offset < size; ++offset) {
int dest = MOD_SIZE(rank + offset);
if (outgoing_counts[dest] != 0) {
while (outgoing_reqs_active[dest]) CHECK_MPI_REQS;
MPI_Isend(&outgoing[dest * coalescing_size * 2], outgoing_counts[dest], MPI_INT64_T, dest, 0, MPI_COMM_WORLD, &outgoing_reqs[dest]);
outgoing_reqs_active[dest] = 1;
outgoing_counts[dest] = 0;
}
while (outgoing_reqs_active[dest]) CHECK_MPI_REQS;
MPI_Isend(&outgoing[dest * coalescing_size * 2], 0, MPI_INT64_T, dest, 0, MPI_COMM_WORLD, &outgoing_reqs[dest]); 
outgoing_reqs_active[dest] = 1;
while (outgoing_reqs_active[dest]) CHECK_MPI_REQS;
}
while (num_ranks_done < size) CHECK_MPI_REQS;
int64_t global_newq_count;
MPI_Allreduce(&newq_count, &global_newq_count, 1, MPI_INT64_T, MPI_SUM, MPI_COMM_WORLD);
if (global_newq_count == 0) break;
{int64_t* temp = oldq; oldq = newq; newq = temp;}
oldq_count = newq_count;
newq_count = 0;
}
#undef CHECK_MPI_REQS
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
