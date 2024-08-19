#include "tile.h"
#include "sort.h"
#include "timer.h"
#include "thd_info.h"
#include "thread_partition.h"
#include "util.h"
static idx_t * p_mkslabptr(
idx_t const * const inds,
idx_t const nnz,
idx_t const nslabs)
{
idx_t * slabs = (idx_t *) calloc(nslabs+1, sizeof(idx_t));
for(idx_t n=0; n < nnz; ++n) {
idx_t const slabid = inds[n] / TILE_SIZES[0];
slabs[1 + slabid] += 1;
}
for(idx_t s=1; s <= nslabs; ++s) {
slabs[s] += slabs[s-1];
}
return slabs;
}
static idx_t p_fill_uniques(
idx_t const * const inds,
idx_t const start,
idx_t const end,
idx_t * const seen,
idx_t * const uniques)
{
idx_t nuniques = 0;
for(idx_t n=start; n < end; ++n) {
idx_t const jj = inds[n];
seen[jj] += 1;
if(seen[jj] == 1) {
uniques[nuniques++] = jj;
}
}
quicksort(uniques, nuniques);
return nuniques;
}
static void p_tile_uniques(
idx_t const start,
idx_t const end,
sptensor_t * const src,
sptensor_t * const dest,
idx_t const mode,
idx_t * const seen,
idx_t * const uniques,
idx_t const nuniques,
idx_t const tsize)
{
idx_t const ntubes = (nuniques / tsize) + (nuniques % tsize != 0);
idx_t * tmkr = (idx_t *) calloc(ntubes+1, sizeof(idx_t));
tmkr[0] = start;
for(idx_t n=0; n < nuniques; ++n) {
tmkr[1+(n / tsize)] += seen[uniques[n]];
}
for(idx_t t=1; t <= ntubes; ++t) {
tmkr[t] += tmkr[t-1];
}
for(idx_t n=0; n < nuniques; ++n) {
seen[uniques[n]] = n;
}
idx_t const * const ind = src->ind[mode];
for(idx_t n=start; n < end; ++n) {
idx_t const index = tmkr[seen[ind[n]] / tsize];
for(idx_t m=0; m < src->nmodes; ++m) {
dest->ind[m][index] = src->ind[m][n];
}
dest->vals[index] = src->vals[n];
tmkr[seen[ind[n]] / tsize] += 1;
}
free(tmkr);
}
static void p_clear_uniques(
idx_t * const seen,
idx_t * const uniques,
idx_t const nuniques)
{
for(idx_t n=0; n < nuniques; ++n) {
seen[uniques[n]] = 0;
uniques[n] = 0;
}
}
static void p_pack_slab(
idx_t const start,
idx_t const end,
sptensor_t * const tt,
sptensor_t * const tt_buf,
idx_t const * const dim_perm,
idx_t * const * const seen,
idx_t * const * const uniques,
idx_t * const nuniques)
{
idx_t const fibmode = dim_perm[1];
idx_t const idxmode = dim_perm[2];
nuniques[fibmode] = p_fill_uniques(tt->ind[fibmode], start, end,
seen[fibmode], uniques[fibmode]);
p_tile_uniques(start, end, tt, tt_buf, fibmode, seen[fibmode],
uniques[fibmode], nuniques[fibmode], TILE_SIZES[1]);
nuniques[idxmode] = p_fill_uniques(tt_buf->ind[idxmode], start, end,
seen[idxmode], uniques[idxmode]);
p_tile_uniques(start, end, tt_buf, tt, idxmode, seen[idxmode],
uniques[idxmode], nuniques[idxmode], TILE_SIZES[2]);
p_clear_uniques(seen[fibmode], uniques[fibmode], nuniques[fibmode]);
p_clear_uniques(seen[idxmode], uniques[idxmode], nuniques[idxmode]);
}
void tt_tile(
sptensor_t * const tt,
idx_t * dim_perm)
{
timer_start(&timers[TIMER_TILE]);
idx_t const nslices = tt->dims[dim_perm[0]];
idx_t const nslabs = (nslices / TILE_SIZES[0]) +
(nslices % TILE_SIZES[0] != 0);
tt_sort(tt, dim_perm[0], dim_perm);
sptensor_t * tt_buf = tt_alloc(tt->nnz, tt->nmodes);
for(idx_t m=0; m < tt->nmodes; ++m) {
tt_buf->dims[m] = tt->dims[m];
}
idx_t * slabptr = p_mkslabptr(tt->ind[dim_perm[0]], tt->nnz, nslabs);
idx_t * seen[MAX_NMODES];
idx_t * uniques[MAX_NMODES];
idx_t nuniques[MAX_NMODES];
for(idx_t m=1; m < tt->nmodes; ++m) {
seen[dim_perm[m]]    = (idx_t *) calloc(tt->dims[dim_perm[m]], sizeof(idx_t));
uniques[dim_perm[m]] = (idx_t *) calloc(tt->dims[dim_perm[m]], sizeof(idx_t));
}
for(idx_t s=0; s < nslabs; ++s) {
idx_t const start = slabptr[s];
idx_t const end = slabptr[s+1];
p_pack_slab(start, end, tt, tt_buf, dim_perm, seen, uniques, nuniques);
}
for(idx_t m=1; m < tt->nmodes; ++m) {
free(seen[dim_perm[m]]);
free(uniques[dim_perm[m]]);
}
tt_free(tt_buf);
free(slabptr);
timer_stop(&timers[TIMER_TILE]);
}
idx_t * tt_densetile(
sptensor_t * const tt,
idx_t const * const tile_dims)
{
timer_start(&timers[TIMER_TILE]);
idx_t const nmodes = tt->nmodes;
idx_t ntiles = 1;
for(idx_t m=0; m < nmodes; ++m) {
ntiles *= tile_dims[m];
}
idx_t tsizes[MAX_NMODES];
for(idx_t m=0; m < nmodes; ++m) {
tsizes[m] = SS_MAX(tt->dims[m] / tile_dims[m], 1);
}
sptensor_t * newtt = tt_alloc(tt->nnz, tt->nmodes);
idx_t * tcounts_global = splatt_malloc((ntiles+1) * sizeof(*tcounts_global));
for(idx_t t=0; t < ntiles+1; ++t) {
tcounts_global[t] = 0;
}
int const nthreads = splatt_omp_get_max_threads();
idx_t * * tcounts_thread = splatt_malloc(
(nthreads+1) * sizeof(*tcounts_thread));
tcounts_thread[nthreads] = tcounts_global;
idx_t * thread_parts = partition_simple(tt->nnz, nthreads);
#pragma omp parallel
{
int const tid = splatt_omp_get_thread_num();
idx_t const nnz_start = thread_parts[tid];
idx_t const nnz_end   = thread_parts[tid+1];
tcounts_thread[tid] = splatt_malloc(ntiles * sizeof(**tcounts_thread));
for(idx_t tile=0; tile < ntiles; ++tile) {
tcounts_thread[tid][tile] = 0;
}
#pragma omp barrier
idx_t * tcounts_local = tcounts_thread[tid+1];
idx_t coord[MAX_NMODES];
for(idx_t x=nnz_start; x < nnz_end; ++x) {
for(idx_t m=0; m < nmodes; ++m) {
coord[m] = SS_MIN(tt->ind[m][x] / tsizes[m], tile_dims[m]-1);
}
idx_t const id = get_tile_id(tile_dims, nmodes, coord);
assert(id < ntiles);
++tcounts_local[id];
}
#pragma omp barrier
#pragma omp single
{
for(idx_t tile=0; tile < ntiles; ++tile) {
for(int thread=0; thread < nthreads; ++thread) {
tcounts_thread[thread+1][tile] += tcounts_thread[thread][tile];
}
if(tile < (ntiles-1)) {
tcounts_thread[0][tile+1] += tcounts_thread[nthreads][tile];
}
}
} 
tcounts_local = tcounts_thread[tid];
for(idx_t x=nnz_start; x < nnz_end; ++x) {
for(idx_t m=0; m < nmodes; ++m) {
coord[m] = SS_MIN(tt->ind[m][x] / tsizes[m], tile_dims[m]-1);
}
idx_t const id = get_tile_id(tile_dims, nmodes, coord);
assert(id < ntiles);
idx_t const newidx = tcounts_local[id]++;
newtt->vals[newidx] = tt->vals[x];
for(idx_t m=0; m < nmodes; ++m) {
newtt->ind[m][newidx] = tt->ind[m][x];
}
}
splatt_free(tcounts_local);
} 
par_memcpy(tt->vals, newtt->vals, tt->nnz * sizeof(*tt->vals));
for(idx_t m=0; m < nmodes; ++m) {
par_memcpy(tt->ind[m], newtt->ind[m], tt->nnz * sizeof(**tt->ind));
}
memmove(tcounts_global+1, tcounts_global, ntiles * sizeof(*tcounts_global));
tcounts_global[0] = 0;
assert(tcounts_global[ntiles] == tt->nnz);
tt_free(newtt);
splatt_free(tcounts_thread);
splatt_free(thread_parts);
timer_stop(&timers[TIMER_TILE]);
return tcounts_global;
}
idx_t get_tile_id(
idx_t const * const tile_dims,
idx_t const nmodes,
idx_t const * const tile_coord)
{
idx_t id = 0;
idx_t mult = 1;
for(idx_t m=nmodes; m-- != 0;) {
id += tile_coord[m] * mult;
mult *= tile_dims[m];
}
if(id >= mult) {
id = TILE_ERR;
}
return id;
}
void fill_tile_coords(
idx_t const * const tile_dims,
idx_t const nmodes,
idx_t const tile_id,
idx_t * const tile_coord)
{
idx_t maxid = 1;
for(idx_t m=0; m < nmodes; ++m) {
maxid *= tile_dims[m];
}
if(tile_id >= maxid) {
for(idx_t m=0; m < nmodes; ++m) {
tile_coord[m] = tile_dims[m];
}
return;
}
idx_t id = tile_id;
for(idx_t m = nmodes; m-- != 0; ) {
tile_coord[m] = id % tile_dims[m];
id /= tile_dims[m];
}
}
idx_t get_next_tileid(
idx_t const previd,
idx_t const * const tile_dims,
idx_t const nmodes,
idx_t const iter_mode,
idx_t const mode_idx)
{
idx_t maxid = 1;
idx_t coords[MAX_NMODES];
for(idx_t m=0; m < nmodes; ++m) {
coords[m] = 0;
maxid *= tile_dims[m];
}
if(previd == TILE_BEGIN) {
coords[iter_mode] = mode_idx;
return get_tile_id(tile_dims, nmodes, coords);
}
if(previd >= maxid) {
return TILE_ERR;
}
fill_tile_coords(tile_dims, nmodes, previd, coords);
idx_t const overmode = (iter_mode == 0) ? 1 : 0;
idx_t pmode = (iter_mode == nmodes-1) ? nmodes-2 : nmodes-1;
++coords[pmode];
while(coords[pmode] == tile_dims[pmode]) {
if(pmode == overmode) {
return TILE_END;
}
coords[pmode] = 0;
--pmode;
if(pmode == iter_mode) {
assert(pmode > 0);
--pmode;
}
++coords[pmode];
}
return get_tile_id(tile_dims, nmodes, coords);
}
