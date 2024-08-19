#include "kt.h"
#include <stdbool.h>
#include <inttypes.h>
#ifndef MAX_NTHREADS
#define MAX_NTHREADS 272
#endif
#ifndef DYNAMIC_CHUNK
#define DYNAMIC_CHUNK 256
#endif
typedef struct
{
int64_t edge_a;
int64_t edge_b;
} tri_edges;
static void p_construct_output(
params_t const * const params,
vault_t * const vault,
int32_t ktmax,
int32_t const * const restrict supports)
{
if (params->outfile == NULL)
return;
int32_t const nvtxs  = vault->ugraph->nvtxs;
ssize_t const * const restrict xadj   = vault->ugraph->xadj;
int32_t const * const restrict adjncy = vault->ugraph->adjncy;
vault->nedges = xadj[nvtxs];
vault->ktmax  = ktmax;
vault->ktedges = gk_malloc(xadj[nvtxs] * sizeof(*vault->ktedges), "ktedges");
#pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
for(int32_t v=0; v < nvtxs; ++v) {
for(ssize_t e = xadj[v]; e < xadj[v+1]; ++e) {
int32_t const v1 = vault->iperm[v];
int32_t const v2 = vault->iperm[adjncy[e]];
vault->ktedges[e].vi = gk_min(v1, v2);
vault->ktedges[e].vj = gk_max(v1, v2);
vault->ktedges[e].k = supports[e];
}
}
}
static int32_t p_intersect_lists(
int32_t * const restrict adj_u,
ssize_t const len_u,
ssize_t const adj_u_offset,
int32_t * const restrict adj_v,
ssize_t const len_v,
ssize_t const adj_v_offset,
tri_edges * const restrict triangles,
int32_t const max_triangles)
{
if(max_triangles == 0) {
return 0;
}
int32_t num_found = 0;
int32_t u_ptr = len_u - 1;
int32_t v_ptr = len_v - 1;
while((u_ptr >= 0) && (v_ptr >= 0)) {
int32_t const u = adj_u[u_ptr];
int32_t const v = adj_v[v_ptr];
if(u < v) {
--v_ptr;
} else if(v < u) {
--u_ptr;
} else {
triangles[num_found].edge_a = u_ptr + adj_u_offset;
triangles[num_found].edge_b = v_ptr + adj_v_offset;
++num_found;
if(num_found == max_triangles) {
return num_found;
}
--u_ptr;
--v_ptr;
}
}
return num_found;
}
static void p_find_triangles(
gk_graph_t const * const lgraph,
gk_graph_t const * const ugraph,
int64_t const * const restrict lgraph_maps,
int32_t const num_triangles,
int32_t const * const restrict supports,
tri_edges     * const restrict triangle_buf,
int32_t       * const restrict h_index,
int32_t const u,
int32_t const v)
{
int32_t found_triangles = 0;
int32_t nnbrs_u = ugraph->xadj[u+1] - ugraph->xadj[u];
int32_t nnbrs_v = ugraph->xadj[v+1] - ugraph->xadj[v];
int32_t * adj_u = &(ugraph->adjncy[ugraph->xadj[u]]);
int32_t * adj_v = &(ugraph->adjncy[ugraph->xadj[v]]);
if(found_triangles != num_triangles) {
int32_t const new_triangles = p_intersect_lists(
adj_u, nnbrs_u, ugraph->xadj[u],
adj_v, nnbrs_v, ugraph->xadj[v],
&(triangle_buf[found_triangles]), num_triangles - found_triangles);
for(int32_t t=0; t < new_triangles; ++t) {
int32_t const uw = supports[triangle_buf[t].edge_a];
int32_t const vw = supports[triangle_buf[t].edge_b];
h_index[t] = gk_min(uw, vw);
}
found_triangles += new_triangles;
}
if(found_triangles != num_triangles) {
nnbrs_v = lgraph->xadj[v+1] - lgraph->xadj[v];
adj_v   = &(lgraph->adjncy[lgraph->xadj[v]]);
int32_t const new_triangles = p_intersect_lists(
adj_u, nnbrs_u, ugraph->xadj[u],
adj_v, nnbrs_v, lgraph->xadj[v],
&(triangle_buf[found_triangles]), num_triangles - found_triangles);
for(int32_t tx=0; tx < new_triangles; ++tx) {
int32_t const t = tx + found_triangles;
triangle_buf[t].edge_b = lgraph_maps[triangle_buf[t].edge_b];
int32_t const uw = supports[triangle_buf[t].edge_a];
int32_t const vw = supports[triangle_buf[t].edge_b];
h_index[t] = gk_min(uw, vw);
}
found_triangles += new_triangles;
}
if(found_triangles != num_triangles) {
nnbrs_u = lgraph->xadj[u+1] - lgraph->xadj[u];
adj_u   = &(lgraph->adjncy[lgraph->xadj[u]]);
int32_t const new_triangles = p_intersect_lists(
adj_u, nnbrs_u, lgraph->xadj[u],
adj_v, nnbrs_v, lgraph->xadj[v],
&(triangle_buf[found_triangles]), num_triangles - found_triangles);
for(int32_t tx=0; tx < new_triangles; ++tx) {
int32_t const t = tx + found_triangles;
triangle_buf[t].edge_a = lgraph_maps[triangle_buf[t].edge_a];
triangle_buf[t].edge_b = lgraph_maps[triangle_buf[t].edge_b];
int32_t const uw = supports[triangle_buf[t].edge_a];
int32_t const vw = supports[triangle_buf[t].edge_b];
h_index[t] = gk_min(uw, vw);
}
found_triangles += new_triangles;
}
assert(found_triangles == num_triangles);
}
static int32_t p_compute_hindex(
int32_t const * const restrict vals,
int32_t       * const restrict buffer,
int32_t const N)
{
for(int32_t i=0; i < N+1; ++i) {
buffer[i] = 0;
}
for(int32_t i=0; i < N; ++i) {
int32_t idx = 0;
if(vals[i] < N) {
idx = vals[i];
} else {
idx = N;
}
++buffer[idx];
}
int32_t sum = 0;
for(int32_t i=N; i >= 0; --i) {
sum += buffer[i];
if(sum >= i) {
return i;
}
}
assert(false);
return -1;
}
static int32_t p_update_edge(
gk_graph_t const * const lgraph,
gk_graph_t const * const ugraph,
int64_t const * const restrict lgraph_maps,
int32_t const * const restrict supports,
tri_edges     * const restrict triangle_buf,
int32_t       * const restrict h_index,
int32_t       * const restrict h_index_buf,
int64_t const edge_idx,
int32_t const u,
char          * const restrict need_update)
{
int32_t const v = ugraph->adjncy[edge_idx];
int32_t const num_triangles = ugraph->iadjwgt[edge_idx];
p_find_triangles(lgraph, ugraph, lgraph_maps, num_triangles, supports,
triangle_buf, h_index, u, v);
int32_t const new_h = p_compute_hindex(h_index, h_index_buf, num_triangles);
if(new_h != supports[edge_idx]) {
for(int32_t t=0; t < num_triangles; ++t) {
need_update[triangle_buf[t].edge_a] = 1;
need_update[triangle_buf[t].edge_b] = 1;
}
}
return new_h;
}
int64_t kt_and(params_t *params, vault_t *vault)
{
printf("THREADS: %d\n", omp_get_max_threads());
gk_startwctimer(vault->timer_tcsetup);
vault->ugraph = kt_PreprocessAndExtractUpper(params, vault);
vault->lgraph = kt_TransposeUforJIK(params, vault->ugraph);
gk_stopwctimer(vault->timer_tcsetup);
int32_t const nvtxs  = vault->ugraph->nvtxs;
int64_t const nedges = vault->ugraph->xadj[nvtxs];
gk_startwctimer(vault->timer_esupport);
vault->ugraph->iadjwgt = gk_i32malloc(nedges, "iadjwgt");
par_memset(vault->ugraph->iadjwgt, 0, nedges * sizeof(*vault->ugraph->iadjwgt));
int32_t * supports     = gk_i32malloc(nedges, "supports");
par_memset(supports, 0, nedges * sizeof(*supports));
int64_t const ntriangles = kt_ComputeEdgeSupport(params, vault);
par_memcpy(supports, vault->ugraph->iadjwgt, nedges * sizeof(*supports));
int64_t const nz_edges = count_nnz(nedges, supports);
gk_stopwctimer(vault->timer_esupport);
printf("Found |V|=%"PRId32" |E|=%"PRId64" |T|=%"PRId64" NZ-SUPPORTS=%"PRId64" (%0.1f%%)\n",
nvtxs, nedges, ntriangles, nz_edges,
100. * (double) nz_edges / (double) nedges);
gk_graph_Free(&vault->lgraph);
vault->lgraph = transpose_graph(vault->ugraph);
int32_t * u_vtxs = gk_malloc(nedges * sizeof(*u_vtxs), "u_vtxs");
#pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
for(int32_t v=0; v < nvtxs; ++v) {
for(ssize_t e = vault->ugraph->xadj[v]; e < vault->ugraph->xadj[v+1]; ++e) {
u_vtxs[e] = v;
}
}
int64_t * lgraph_maps = gk_malloc(nedges * sizeof(*lgraph_maps),
"lgraph_maps");
#pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK)
for(int32_t v=0; v < nvtxs; ++v) {
for(ssize_t le = vault->lgraph->xadj[v]; le < vault->lgraph->xadj[v+1];
++le) {
int32_t const u = vault->lgraph->adjncy[le];
for(ssize_t ue = vault->ugraph->xadj[u]; ue < vault->ugraph->xadj[u+1];
++ue) {
if(vault->ugraph->adjncy[ue] == v) {
lgraph_maps[le] = ue;
}
}
}
}
int32_t const max_support = max_elem(supports, nedges);
int32_t * h_index[MAX_NTHREADS];
int32_t * h_index_buf[MAX_NTHREADS];
tri_edges * triangle_buf[MAX_NTHREADS];
#pragma omp parallel
{
int const tid = omp_get_thread_num();
h_index[tid] = gk_malloc(max_support * sizeof(**h_index), "h_index");
h_index_buf[tid] = gk_malloc((max_support+1) * sizeof(**h_index),
"h_index_buf");
triangle_buf[tid] = gk_malloc(max_support * sizeof(**triangle_buf),
"tri_buf");
}
char * need_update = gk_malloc(nedges * sizeof(*need_update), "need_update");
char * need_update_new = gk_malloc(nedges * sizeof(*need_update),
"need_update_new");
#pragma omp parallel for schedule(static)
for(int64_t e=0; e < nedges; ++e) {
need_update[e]     = 1;
need_update_new[e] = 0;
}
gk_startwctimer(vault->timer_ktpeeling);
bool done = false;
while(!done) {
int64_t nchanges = 0;
#pragma omp parallel for schedule(dynamic, DYNAMIC_CHUNK) reduction(+: nchanges)
for(int64_t e=0; e < nedges; ++e) {
if(!need_update[e]) {
continue;
}
int const tid = omp_get_thread_num();
int32_t const new_support = p_update_edge(vault->lgraph, vault->ugraph,
lgraph_maps, supports, triangle_buf[tid], h_index[tid], h_index_buf[tid],
e, u_vtxs[e], need_update_new);
if(supports[e] != new_support) {
supports[e] = new_support;
++nchanges;
}
need_update[e] = 0;
} 
printf("changes: %"PRId64"\n", nchanges);
done = (nchanges == 0);
char * tmp = need_update_new;
need_update_new = need_update;
need_update = tmp;
} 
gk_stopwctimer(vault->timer_ktpeeling);
int32_t max_ktruss = 0;
#pragma omp parallel for reduction(max: max_ktruss)
for(int64_t e=0; e < nedges; ++e) {
supports[e] += 2;
max_ktruss = gk_max(max_ktruss, supports[e]);
}
printf("\nMAX K-TRUSS: %d\n\n", max_ktruss);
#pragma omp parallel
{
int const tid = omp_get_thread_num();
gk_free((void **) &h_index[tid], LTERM);
gk_free((void **) &h_index_buf[tid], LTERM);
gk_free((void **) &triangle_buf[tid], LTERM);
}
p_construct_output(params, vault, max_ktruss, supports);
gk_free((void **) &need_update, LTERM);
gk_free((void **) &need_update_new, LTERM);
gk_free((void **) &lgraph_maps, LTERM);
gk_free((void **) &supports, LTERM);
gk_free((void **) &u_vtxs, LTERM);
return (int64_t) ntriangles;
}
