#include "csf.h"
#include "sort.h"
#include "tile.h"
#include "util.h"
#include "thread_partition.h"
#include "io.h"
int splatt_csf_load(
char const * const fname,
splatt_idx_t * nmodes,
splatt_csf ** tensors,
double const * const options)
{
sptensor_t * tt = tt_read(fname);
if(tt == NULL) {
return SPLATT_ERROR_BADINPUT;
}
tt_remove_empty(tt);
*tensors = csf_alloc(tt, options);
*nmodes = tt->nmodes;
tt_free(tt);
return SPLATT_SUCCESS;
}
int splatt_csf_convert(
splatt_idx_t const nmodes,
splatt_idx_t const nnz,
splatt_idx_t ** const inds,
splatt_val_t * const vals,
splatt_csf ** tensors,
double const * const options)
{
sptensor_t tt;
tt_fill(&tt, nnz, nmodes, inds, vals);
tt_remove_empty(&tt);
*tensors = csf_alloc(&tt, options);
return SPLATT_SUCCESS;
}
void splatt_free_csf(
splatt_csf * tensors,
double const * const options)
{
csf_free(tensors, options);
}
idx_t p_csf_count_nnz(
idx_t * * fptr,
idx_t const nmodes,
idx_t depth,
idx_t const fiber)
{
if(depth == nmodes-1) {
return 1;
}
idx_t left = fptr[depth][fiber];
idx_t right = fptr[depth][fiber+1];
++depth;
for(; depth < nmodes-1; ++depth) {
left = fptr[depth][left];
right = fptr[depth][right];
}
return right - left;
}
static void p_order_dims_small(
idx_t const * const dims,
idx_t const nmodes,
idx_t * const perm_dims)
{
idx_t sorted[MAX_NMODES];
idx_t matched[MAX_NMODES];
for(idx_t m=0; m < nmodes; ++m) {
sorted[m] = dims[m];
matched[m] = 0;
}
quicksort(sorted, nmodes);
for(idx_t mfind=0; mfind < nmodes; ++mfind) {
for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
perm_dims[mfind] = mcheck;
matched[mcheck] = 1;
break;
}
}
}
}
static void p_order_dims_inorder(
idx_t const * const dims,
idx_t const nmodes,
idx_t const custom_mode,
idx_t * const perm_dims)
{
for(idx_t m=0; m < nmodes; ++m) {
perm_dims[m] = m;
}
for(idx_t m=0; m < nmodes; ++m) {
if(perm_dims[m] == custom_mode) {
memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
perm_dims[0] = custom_mode;
break;
}
}
}
static void p_order_dims_minusone(
idx_t const * const dims,
idx_t const nmodes,
idx_t const custom_mode,
idx_t * const perm_dims)
{
p_order_dims_small(dims, nmodes, perm_dims);
for(idx_t m=0; m < nmodes; ++m) {
if(perm_dims[m] == custom_mode) {
memmove(perm_dims + 1, perm_dims, (m) * sizeof(m));
perm_dims[0] = custom_mode;
break;
}
}
}
static void p_order_dims_large(
idx_t const * const dims,
idx_t const nmodes,
idx_t * const perm_dims)
{
idx_t sorted[MAX_NMODES];
idx_t matched[MAX_NMODES];
for(idx_t m=0; m < nmodes; ++m) {
sorted[m] = dims[m];
matched[m] = 0;
}
quicksort(sorted, nmodes);
for(idx_t m=0; m < nmodes/2; ++m) {
idx_t tmp = sorted[nmodes-m-1];
sorted[nmodes-m-1] = sorted[m];
sorted[m] = tmp;
}
for(idx_t mfind=0; mfind < nmodes; ++mfind) {
for(idx_t mcheck=0; mcheck < nmodes; ++mcheck) {
if(sorted[mfind] == dims[mcheck] && !matched[mcheck]) {
perm_dims[mfind] = mcheck;
matched[mcheck] = 1;
break;
}
}
}
}
static void p_mk_outerptr(
splatt_csf * const ct,
sptensor_t const * const tt,
idx_t const tile_id,
idx_t const * const nnztile_ptr)
{
idx_t const nnzstart = nnztile_ptr[tile_id];
idx_t const nnzend   = nnztile_ptr[tile_id+1];
assert(nnzstart < nnzend);
idx_t const nnz = nnzend - nnzstart;
csf_sparsity * const pt = ct->pt + tile_id;
idx_t const * const restrict ttind =
nnzstart + tt->ind[csf_depth_to_mode(ct, 0)];
int const nthreads = splatt_omp_get_max_threads();
idx_t * thread_parts = partition_simple(nnz, nthreads);
idx_t * thread_nfibs = splatt_malloc((nthreads+1) * sizeof(*thread_nfibs));
thread_nfibs[0] = 1;
#pragma omp parallel
{
int const tid = splatt_omp_get_thread_num();
idx_t const nnz_start = SS_MAX(thread_parts[tid], 1); 
idx_t const nnz_end = thread_parts[tid+1];
idx_t local_nfibs = 0;
for(idx_t x=nnz_start; x < nnz_end; ++x) {
assert(ttind[x-1] <= ttind[x]);
if(ttind[x] != ttind[x-1]) {
++local_nfibs;
}
}
thread_nfibs[tid+1] = local_nfibs; 
#pragma omp barrier
#pragma omp single
{
for(int t=0; t < nthreads; ++t) {
thread_nfibs[t+1] += thread_nfibs[t];
}
idx_t const nfibs = thread_nfibs[nthreads];
ct->pt[tile_id].nfibs[0] = nfibs;
assert(nfibs <= ct->dims[csf_depth_to_mode(ct, 0)]);
pt->fptr[0] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
if((ct->ntiles > 1) || (tt->dims[csf_depth_to_mode(ct, 0)] != nfibs)) {
pt->fids[0] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
pt->fids[0][0] = ttind[0];
} else {
pt->fids[0] = NULL;
}
pt->fptr[0][0] = 0;
pt->fptr[0][nfibs] = nnz;
} 
idx_t  * const restrict fp = pt->fptr[0];
idx_t  * const restrict fi = pt->fids[0];
idx_t nfound = thread_nfibs[tid];
if(fi == NULL) {
for(idx_t n=nnz_start; n < nnz_end; ++n) {
if(ttind[n] != ttind[n-1]) {
fp[nfound++] = n;
}
}
} else {
for(idx_t n=nnz_start; n < nnz_end; ++n) {
if(ttind[n] != ttind[n-1]) {
fi[nfound] = ttind[n];
fp[nfound++] = n;
}
}
}
} 
splatt_free(thread_parts);
splatt_free(thread_nfibs);
}
static void p_mk_fptr(
splatt_csf * const ct,
sptensor_t const * const tt,
idx_t const tile_id,
idx_t const * const nnztile_ptr,
idx_t const mode)
{
assert(mode < ct->nmodes);
idx_t const nnzstart = nnztile_ptr[tile_id];
idx_t const nnzend   = nnztile_ptr[tile_id+1];
idx_t const nnz = nnzend - nnzstart;
if(mode == 0) {
p_mk_outerptr(ct, tt, tile_id, nnztile_ptr);
return;
}
idx_t const * const restrict ttind =
nnzstart + tt->ind[csf_depth_to_mode(ct, mode)];
csf_sparsity * const pt = ct->pt + tile_id;
idx_t * const restrict fprev = pt->fptr[mode-1];
int const nthreads = splatt_omp_get_max_threads();
idx_t * thread_parts = partition_simple(pt->nfibs[mode-1], nthreads);
idx_t * thread_nfibs = splatt_malloc((nthreads+1) * sizeof(*thread_nfibs));
thread_nfibs[0] = 0;
#pragma omp parallel
{
int const tid = splatt_omp_get_thread_num();
idx_t const slice_start = thread_parts[tid];
idx_t const slice_end   = thread_parts[tid+1];
idx_t local_nfibs = 0;
for(idx_t s=slice_start; s < slice_end; ++s) {
++local_nfibs; 
for(idx_t f=fprev[s]+1; f < fprev[s+1]; ++f) {
if(ttind[f] != ttind[f-1]) {
++local_nfibs;
}
}
}
thread_nfibs[tid+1] = local_nfibs; 
idx_t const fprev_end = fprev[slice_end];
#pragma omp barrier
#pragma omp single
{
for(int t=0; t < nthreads; ++t) {
thread_nfibs[t+1] += thread_nfibs[t];
}
idx_t const nfibs = thread_nfibs[nthreads];
pt->nfibs[mode] = nfibs;
pt->fptr[mode] = splatt_malloc((nfibs+1) * sizeof(**(pt->fptr)));
pt->fptr[mode][0] = 0;
pt->fids[mode] = splatt_malloc(nfibs * sizeof(**(pt->fids)));
} 
idx_t * const restrict fp = pt->fptr[mode];
idx_t * const restrict fi = pt->fids[mode];
idx_t nfound = thread_nfibs[tid];
for(idx_t s=slice_start; s < slice_end; ++s) {
idx_t const start = fprev[s]+1;
idx_t const end = (s == slice_end - 1) ? fprev_end : fprev[s+1];
fprev[s] = nfound;
fi[nfound] = ttind[start-1];
fp[nfound++] = start-1;
for(idx_t f=start; f < end; ++f) {
if(ttind[f] != ttind[f-1]) {
fi[nfound] = ttind[f];
fp[nfound++] = f;
}
}
}
if(tid == nthreads - 1) {
fprev[pt->nfibs[mode-1]] = thread_nfibs[nthreads];
fp[thread_nfibs[nthreads]] = nnz;
}
} 
splatt_free(thread_parts);
splatt_free(thread_nfibs);
}
static void p_csf_alloc_untiled(
splatt_csf * const ct,
sptensor_t * const tt)
{
idx_t const nmodes = tt->nmodes;
tt_sort(tt, ct->dim_perm[0], ct->dim_perm);
ct->ntiles = 1;
ct->ntiled_modes = 0;
for(idx_t m=0; m < nmodes; ++m) {
ct->tile_dims[m] = 1;
}
ct->pt = splatt_malloc(sizeof(*(ct->pt)));
csf_sparsity * const pt = ct->pt;
pt->nfibs[nmodes-1] = ct->nnz;
pt->fids[nmodes-1] = splatt_malloc(ct->nnz * sizeof(**(pt->fids)));
pt->vals           = splatt_malloc(ct->nnz * sizeof(*(pt->vals)));
par_memcpy(pt->fids[nmodes-1], tt->ind[csf_depth_to_mode(ct, nmodes-1)],
ct->nnz * sizeof(**(pt->fids)));
par_memcpy(pt->vals, tt->vals, ct->nnz * sizeof(*(pt->vals)));
idx_t nnz_ptr[2];
nnz_ptr[0] = 0;
nnz_ptr[1] = tt->nnz;
for(idx_t m=0; m < tt->nmodes-1; ++m) {
p_mk_fptr(ct, tt, 0, nnz_ptr, m);
}
}
static void p_csf_alloc_densetile(
splatt_csf * const ct,
sptensor_t * const tt,
double const * const splatt_opts)
{
idx_t const nmodes = tt->nmodes;
ct->ntiled_modes = (idx_t)splatt_opts[SPLATT_OPTION_TILELEVEL];
ct->ntiled_modes = SS_MIN(ct->ntiled_modes, ct->nmodes);
idx_t const tile_depth = ct->nmodes - ct->ntiled_modes;
idx_t ntiles = 1;
for(idx_t m=0; m < nmodes; ++m) {
idx_t const depth = csf_mode_to_depth(ct, m);
if(depth >= tile_depth) {
ct->tile_dims[m] = (idx_t) splatt_opts[SPLATT_OPTION_NTHREADS];
} else {
ct->tile_dims[m] = 1;
}
ntiles *= ct->tile_dims[m];
}
tt_sort(tt, ct->dim_perm[0], ct->dim_perm);
idx_t * nnz_ptr = tt_densetile(tt, ct->tile_dims);
ct->ntiles = ntiles;
ct->pt = splatt_malloc(ntiles * sizeof(*(ct->pt)));
for(idx_t t=0; t < ntiles; ++t) {
idx_t const startnnz = nnz_ptr[t];
idx_t const endnnz   = nnz_ptr[t+1];
idx_t const ptnnz = endnnz - startnnz;
csf_sparsity * const pt = ct->pt + t;
if(ptnnz == 0) {
for(idx_t m=0; m < ct->nmodes; ++m) {
pt->fptr[m] = NULL;
pt->fids[m] = NULL;
pt->nfibs[m] = 0;
}
pt->fptr[0] = (idx_t *) splatt_malloc(2 * sizeof(**(pt->fptr)));
pt->fptr[0][0] = 0;
pt->fptr[0][1] = 0;
pt->vals = NULL;
continue;
}
idx_t const leaves = nmodes-1;
pt->nfibs[leaves] = ptnnz;
pt->fids[leaves] = splatt_malloc(ptnnz * sizeof(**(pt->fids)));
par_memcpy(pt->fids[leaves], tt->ind[csf_depth_to_mode(ct, leaves)] + startnnz,
ptnnz * sizeof(**(pt->fids)));
pt->vals = splatt_malloc(ptnnz * sizeof(*(pt->vals)));
par_memcpy(pt->vals, tt->vals + startnnz, ptnnz * sizeof(*(pt->vals)));
for(idx_t m=0; m < leaves; ++m) {
p_mk_fptr(ct, tt, t, nnz_ptr, m);
}
}
splatt_free(nnz_ptr);
}
static void p_fill_dim_iperm(
splatt_csf * const ct)
{
for(idx_t level=0; level < ct->nmodes; ++level) {
ct->dim_iperm[ct->dim_perm[level]] = level;
}
}
static void p_mk_csf(
splatt_csf * const ct,
sptensor_t * const tt,
csf_mode_type mode_type,
idx_t const mode,
double const * const splatt_opts)
{
ct->nnz = tt->nnz;
ct->nmodes = tt->nmodes;
for(idx_t m=0; m < tt->nmodes; ++m) {
ct->dims[m] = tt->dims[m];
}
csf_find_mode_order(tt->dims, tt->nmodes, mode_type, mode, ct->dim_perm);
p_fill_dim_iperm(ct);
ct->which_tile = splatt_opts[SPLATT_OPTION_TILE];
switch(ct->which_tile) {
case SPLATT_NOTILE:
p_csf_alloc_untiled(ct, tt);
break;
case SPLATT_DENSETILE:
p_csf_alloc_densetile(ct, tt, splatt_opts);
break;
default:
fprintf(stderr, "SPLATT: tiling '%d' unsupported for CSF tensors.\n",
ct->which_tile);
break;
}
}
void csf_free(
splatt_csf * const csf,
double const * const opts)
{
idx_t ntensors = 0;
splatt_csf_type which = opts[SPLATT_OPTION_CSF_ALLOC];
switch(which) {
case SPLATT_CSF_ONEMODE:
ntensors = 1;
break;
case SPLATT_CSF_TWOMODE:
ntensors = 2;
break;
case SPLATT_CSF_ALLMODE:
ntensors = csf[0].nmodes;
break;
}
for(idx_t i=0; i < ntensors; ++i) {
csf_free_mode(csf + i);
}
free(csf);
}
void csf_free_mode(
splatt_csf * const csf)
{
for(idx_t t=0; t < csf->ntiles; ++t) {
free(csf->pt[t].vals);
free(csf->pt[t].fids[csf->nmodes-1]);
for(idx_t m=0; m < csf->nmodes-1; ++m) {
free(csf->pt[t].fptr[m]);
free(csf->pt[t].fids[m]);
}
}
free(csf->pt);
}
void csf_find_mode_order(
idx_t const * const dims,
idx_t const nmodes,
csf_mode_type which,
idx_t const mode,
idx_t * const perm_dims)
{
switch(which) {
case CSF_SORTED_SMALLFIRST:
p_order_dims_small(dims, nmodes, perm_dims);
break;
case CSF_SORTED_BIGFIRST:
p_order_dims_large(dims, nmodes, perm_dims);
break;
case CSF_INORDER_MINUSONE:
p_order_dims_inorder(dims, nmodes, mode, perm_dims);
break;
case CSF_SORTED_MINUSONE:
p_order_dims_minusone(dims, nmodes, mode, perm_dims);
break;
case CSF_MODE_CUSTOM:
break;
default:
fprintf(stderr, "SPLATT: csf_mode_type '%d' not recognized.\n", which);
break;
}
}
size_t csf_storage(
splatt_csf const * const tensors,
double const * const opts)
{
idx_t ntensors = 0;
splatt_csf_type which_alloc = opts[SPLATT_OPTION_CSF_ALLOC];
switch(which_alloc) {
case SPLATT_CSF_ONEMODE:
ntensors = 1;
break;
case SPLATT_CSF_TWOMODE:
ntensors = 2;
break;
case SPLATT_CSF_ALLMODE:
ntensors = tensors[0].nmodes;
break;
}
size_t bytes = 0;
for(idx_t m=0; m < ntensors; ++m) {
splatt_csf const * const ct = tensors + m;
bytes += ct->nnz * sizeof(*(ct->pt->vals)); 
bytes += ct->nnz * sizeof(**(ct->pt->fids)); 
bytes += ct->ntiles * sizeof(*(ct->pt)); 
for(idx_t t=0; t < ct->ntiles; ++t) {
csf_sparsity const * const pt = ct->pt + t;
for(idx_t m=0; m < ct->nmodes-1; ++m) {
bytes += (pt->nfibs[m]+1) * sizeof(**(pt->fptr)); 
if(pt->fids[m] != NULL) {
bytes += pt->nfibs[m] * sizeof(**(pt->fids)); 
}
}
}
}
return bytes;
}
splatt_csf * csf_alloc(
sptensor_t * const tt,
double const * const opts)
{
splatt_csf * ret = NULL;
double * tmp_opts = NULL;
idx_t last_mode = 0;
int tmp = 0;
switch((splatt_csf_type) opts[SPLATT_OPTION_CSF_ALLOC]) {
case SPLATT_CSF_ONEMODE:
ret = splatt_malloc(sizeof(*ret));
p_mk_csf(ret, tt, CSF_SORTED_SMALLFIRST, 0, opts);
break;
case SPLATT_CSF_TWOMODE:
ret = splatt_malloc(2 * sizeof(*ret));
p_mk_csf(ret + 0, tt, CSF_SORTED_SMALLFIRST, 0, opts);
tmp_opts = splatt_default_opts();
memcpy(tmp_opts, opts, SPLATT_OPTION_NOPTIONS * sizeof(*opts));
tmp_opts[SPLATT_OPTION_TILE] = SPLATT_NOTILE;
last_mode = csf_depth_to_mode(&(ret[0]), tt->nmodes-1);
p_mk_csf(ret + 1, tt, CSF_SORTED_MINUSONE, last_mode, tmp_opts);
free(tmp_opts);
break;
case SPLATT_CSF_ALLMODE:
ret = splatt_malloc(tt->nmodes * sizeof(*ret));
for(idx_t m=0; m < tt->nmodes; ++m) {
p_mk_csf(ret + m, tt, CSF_SORTED_MINUSONE, m, opts);
}
break;
}
return ret;
}
void csf_alloc_mode(
sptensor_t * const tt,
csf_mode_type which_ordering,
idx_t const mode_special,
splatt_csf * const csf,
double const * const opts)
{
p_mk_csf(csf, tt, which_ordering, mode_special, opts);
}
val_t csf_frobsq(
splatt_csf const * const tensor)
{
double norm = 0;
#pragma omp parallel reduction(+:norm)
{
for(idx_t t=0; t < tensor->ntiles; ++t) {
val_t const * const vals = tensor->pt[t].vals;
if(vals == NULL) {
continue;
}
idx_t const nnz = tensor->pt[t].nfibs[tensor->nmodes-1];
#pragma omp for schedule(static) nowait
for(idx_t n=0; n < nnz; ++n) {
norm += vals[n] * vals[n];
}
}
} 
return (val_t) norm;
}
idx_t * csf_partition_1d(
splatt_csf const * const csf,
idx_t const tile_id,
idx_t const nparts)
{
idx_t const nslices = csf->pt[tile_id].nfibs[0];
idx_t * weights = splatt_malloc(nslices * sizeof(*weights));
#pragma omp parallel for schedule(static)
for(idx_t i=0; i < nslices; ++i) {
weights[i] = p_csf_count_nnz(csf->pt[tile_id].fptr, csf->nmodes, 0, i);
}
idx_t bneck;
idx_t * parts = partition_weighted(weights, nslices, nparts, &bneck);
splatt_free(weights);
return parts;
}
idx_t * csf_partition_tiles_1d(
splatt_csf const * const csf,
idx_t const nparts)
{
idx_t const nmodes = csf->nmodes;
idx_t const ntiles = csf->ntiles;
idx_t * weights = splatt_malloc(ntiles * sizeof(*weights));
#pragma omp parallel for schedule(static)
for(idx_t i=0; i < ntiles; ++i) {
weights[i] = csf->pt[i].nfibs[nmodes-1];
}
idx_t bneck;
idx_t * parts = partition_weighted(weights, ntiles, nparts, &bneck);
splatt_free(weights);
return parts;
}
