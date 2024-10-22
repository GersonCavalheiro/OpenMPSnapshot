#include "kt_sbucket.h"
void sbucket_update_edge(
support_bucket_t * const sbucket,
int64_t const edge_id,
int32_t const support,
int32_t const ktruss)
{
slist_s * const slist = sbucket->slist;
if(support < 0 || support == slist[edge_id].support) {
return;
}
int32_t const min_sup = ktruss - 3;
ssize_t * shead = sbucket->list_head;
int32_t const old_sup = gk_max(slist[edge_id].support, min_sup);
if(shead[old_sup] == edge_id) {
shead[old_sup] = slist[edge_id].next_eid;
slist[slist[edge_id].next_eid].prev_eid = -1;
} else {
slist[slist[edge_id].prev_eid].next_eid = slist[edge_id].next_eid;
slist[slist[edge_id].next_eid].prev_eid = slist[edge_id].prev_eid;
}
int32_t const new_sup = gk_max(support, min_sup);
slist[edge_id].support  = support;
slist[edge_id].prev_eid = -1;
slist[edge_id].next_eid = shead[new_sup];
slist[shead[new_sup]].prev_eid = edge_id;
shead[new_sup] = edge_id;
}
int64_t sbucket_count_support_size(
support_bucket_t const * const sbucket,
int32_t const support)
{
if(support >= sbucket->nsupports) {
return 0;
}
int64_t nedges = 0;
ssize_t e_id = sbucket->list_head[support];
while(e_id != -1) {
++nedges;
e_id = sbucket->slist[e_id].next_eid;
}
return nedges;
}
int64_t sbucket_get_frontier(
support_bucket_t * const sbuckets,
int32_t const support,
int64_t * frontier)
{
int const nbuckets = omp_get_max_threads() * KT_BUCKETS_PER_THREAD;
int64_t bucket_sizes[1 + KT_MAX_THREADS * KT_BUCKETS_PER_THREAD];
#pragma omp parallel
{
#pragma omp for schedule(dynamic, 1)
for(int b=0; b < nbuckets; ++b) {
bucket_sizes[b] = sbucket_count_support_size(&(sbuckets[b]), support);
}
#pragma omp single
{
int b;
MAKECSR(b, nbuckets, bucket_sizes);
}
#pragma omp for schedule(dynamic, 1)
for(int b=0; b < nbuckets; ++b) {
int64_t * buffer = &(frontier[bucket_sizes[b]]);
int64_t edge_ptr = 0;
ssize_t e_id = sbuckets[b].list_head[support];
while(e_id != -1) {
buffer[edge_ptr++] = e_id;
e_id = sbuckets[b].slist[e_id].next_eid;
}
sbuckets[b].list_head[support] = -1;
sbuckets[b].slist[-1].prev_eid = -1;
sbuckets[b].slist[-1].next_eid = -1;
} 
} 
return bucket_sizes[nbuckets];
}
void sbucket_fill_edges(
support_bucket_t const * const sbucket,
int32_t const support,
int64_t * const restrict edge_ids)
{
if(support >= sbucket->nsupports) {
return;
}
int64_t edge_ptr = 0;
ssize_t e_id = sbucket->list_head[support];
while(e_id != -1) {
edge_ids[edge_ptr++] = e_id;
e_id = sbucket->slist[e_id].next_eid;
}
}
support_bucket_t * sbucket_alloc(
edge_t const * const edges,
int32_t const * const supports,
int64_t const global_nedges,
thread_ws * * thd_ws)
{
int const nbuckets = omp_get_max_threads() * KT_BUCKETS_PER_THREAD;
support_bucket_t * sbuckets = gk_malloc(nbuckets * sizeof(*sbuckets),
"sbuckets");
int32_t const nsupports = max_elem(supports, global_nedges) + 1;
slist_s * big_slist = gk_malloc((global_nedges+1)*sizeof(*big_slist),
"big_slist");
par_memset(big_slist, 0, (global_nedges+1) * sizeof(*big_slist));
++big_slist; 
#pragma omp parallel for schedule(static, 1)
for(int bucket=0; bucket < nbuckets; ++bucket) {
support_bucket_t * sbucket = &(sbuckets[bucket]);
sbucket->nsupports = nsupports;
sbucket->nowned_edges = 0;
sbucket->slist = big_slist;
sbucket->list_head =
gk_malloc(sbucket->nsupports * sizeof(*sbucket->list_head), "list_head");
ssize_t * const shead = sbucket->list_head;
for(int32_t s=0; s < sbucket->nsupports; ++s) {
shead[s] = -1;
}
}
for(int64_t e=0; e < global_nedges; ++e) {
int64_t const bucket = map_edge_to_bucket(e, thd_ws[0]);
support_bucket_t * sbucket = &(sbuckets[bucket]);
slist_s * slist = sbucket->slist;
ssize_t * const shead = sbucket->list_head;
int32_t const sup = supports[e];
slist[e].prev_eid = -1;
slist[e].next_eid = shead[sup];
slist[e].support  = sup;
if(shead[sup] != -1) {
slist[shead[sup]].prev_eid = e;
}
shead[sup] = e;
++sbucket->nowned_edges;
} 
return sbuckets;
}
void sbucket_free(
support_bucket_t * sbucket)
{
--(sbucket->slist);
gk_free((void **) &(sbucket->slist), LTERM);
gk_free((void **) &sbucket->list_head, LTERM);
gk_free((void **) &sbucket, LTERM);
}
