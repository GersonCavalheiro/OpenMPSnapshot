#include "tc.h"
#define SBSIZE  64
#define DBSIZE  8
gk_graph_t *ptc_Preprocess(params_t *params, vault_t *vault)
{
int32_t vi, nvtxs, nthreads, maxdegree=0, csrange=0;
ssize_t *xadj, *nxadj, *psums;
int32_t *adjncy, *nadjncy, *perm=NULL, *iperm=NULL, *chunkptr=NULL;
int32_t *gcounts;
gk_graph_t *graph;
nthreads = params->nthreads;
nvtxs  = vault->graph->nvtxs;
xadj   = vault->graph->xadj;
adjncy = vault->graph->adjncy;
graph = gk_graph_Create();
graph->nvtxs  = nvtxs;
graph->xadj   = nxadj = gk_zmalloc(nvtxs+1, "nxadj");
graph->adjncy = nadjncy = gk_i32malloc(nvtxs+xadj[nvtxs], "nadjncy");
perm  = gk_i32malloc(nvtxs, "perm");   
iperm = gk_i32malloc(nvtxs, "iperm");  
#pragma omp parallel for schedule(static,4096) default(none) shared(nvtxs, xadj) reduction(max: maxdegree)
for (vi=0; vi<nvtxs; vi++) 
maxdegree = gk_max(maxdegree, (int32_t)(xadj[vi+1]-xadj[vi]));
csrange = maxdegree+1; 
csrange = 16*((csrange+15)/16); 
gcounts = gk_i32malloc(nthreads*csrange, "gcounts");
psums   = gk_zmalloc(nthreads, "psums");
#pragma omp parallel default(none) shared(vault, nvtxs, nthreads, maxdegree, csrange, xadj, adjncy, nxadj, nadjncy, perm, iperm, gcounts, psums, chunkptr, stdout) 
{
int32_t vi, vistart, viend, vj, nedges, nchunks;
int32_t ti, di, ci, dstart, dend;
int32_t *counts, *buffer;
ssize_t ej, ejend, psum, chunksize;
#if defined(_OPENMP)
int mytid = omp_get_thread_num();
#else
int mytid = 0;
#endif
vistart = mytid*((nvtxs+nthreads-1)/nthreads);
viend   = gk_min(nvtxs, (mytid+1)*((nvtxs+nthreads-1)/nthreads));
dstart = mytid*((csrange+nthreads-1)/nthreads);
dend   = gk_min(csrange, (mytid+1)*((csrange+nthreads-1)/nthreads));
counts = gcounts + mytid*csrange;
gk_i32set(csrange, 0, counts);
for (vi=vistart; vi<viend; vi++) 
counts[xadj[vi+1]-xadj[vi]]++;
#pragma omp barrier
for (psum=0, ti=0; ti<nthreads; ti++) {
counts = gcounts + ti*csrange;
for (di=dstart; di<dend; di++) 
psum += counts[di];
}
psums[mytid] = psum;
#pragma omp barrier
#pragma omp single
for (ti=1; ti<nthreads; ti++)
psums[ti] += psums[ti-1];
#pragma omp barrier
psum = psums[mytid];
for (di=dend-1; di>=dstart; di--) { 
counts = gcounts + (nthreads-1)*csrange;
for (ti=nthreads-1; ti>=0; ti--) {
psum -= counts[di];
counts[di] = psum;
counts -= csrange;
}
}
#pragma omp barrier
counts = gcounts + mytid*csrange;
for (vi=vistart; vi<viend; vi++) {
perm[vi] = counts[xadj[vi+1]-xadj[vi]]++;
nxadj[perm[vi]] = xadj[vi+1]-xadj[vi]+1; 
iperm[perm[vi]] = vi;
}
#pragma omp barrier
#pragma omp barrier
for (psum=0, vi=vistart; vi<viend; vi++)
psum += nxadj[vi];
psums[mytid] = psum;
#pragma omp barrier
#pragma omp single
for (ti=1; ti<nthreads; ti++)
psums[ti] += psums[ti-1];
#pragma omp barrier
psum = psums[mytid];
if (mytid == nthreads-1)
nxadj[nvtxs] = psum;
for (vi=viend-1; vi>=vistart; vi--) { 
psum -= nxadj[vi];
nxadj[vi] = psum;
}
#pragma omp barrier
chunksize = 1+psums[nthreads-1]/(100*nthreads);
for (nchunks=0, psum=0, vi=vistart; vi<viend; vi++) {
if ((psum += nxadj[vi+1]-nxadj[vi]) >= chunksize) {
nchunks++;
psum = 0;
}
}
psums[mytid] = nchunks+1;
#pragma omp barrier
#pragma omp single
for (ti=1; ti<nthreads; ti++)
psums[ti] += psums[ti-1];
#pragma omp barrier
#pragma omp single
chunkptr = gk_i32malloc(psums[nthreads-1]+1, "chunkptr");
#pragma omp barrier
nchunks = psums[mytid];
chunkptr[nchunks] = viend;
for (psum=0, vi=viend-1; vi>=vistart; vi--) {
if ((psum += nxadj[vi+1]-nxadj[vi]) >= chunksize) {
chunkptr[--nchunks] = vi;
psum = 0;
}
}
if (mytid == 0)
chunkptr[0] = 0;
#pragma omp barrier
nchunks = psums[nthreads-1]; 
#pragma omp for schedule(dynamic, 1) nowait
for (ci=nchunks-1; ci>=0; ci--) {
for (vi=chunkptr[ci]; vi<chunkptr[ci+1]; vi++) {
vj = iperm[vi];
buffer = nadjncy+nxadj[vi];
for (nedges=0, ej=xadj[vj], ejend=xadj[vj+1]; ej<ejend; ej++, nedges++) 
buffer[nedges] = perm[adjncy[ej]];
buffer[nedges++] = vi; 
if (nedges > 1)
gk_i32sorti(nedges, buffer);  
}
}
}
gk_free((void **)&perm, &iperm, &gcounts, &psums, &chunkptr, LTERM);
return graph;
}
int64_t ptc_MapJIK(params_t *params, vault_t *vault)
{
int32_t vi, vj, nvtxs, startv;
ssize_t ei, ej;
int64_t ntriangles=0;
ssize_t *xadj, *uxadj;
int32_t *adjncy;
int32_t l2, maxhmsize=0;
gk_graph_t *graph;
uint64_t nprobes=0;
gk_startwctimer(vault->timer_pp);
graph = ptc_Preprocess(params, vault);
gk_stopwctimer(vault->timer_pp);
nvtxs  = graph->nvtxs;
xadj   = graph->xadj;
adjncy = graph->adjncy;
uxadj = gk_zmalloc(nvtxs, "uxadj"); 
gk_startwctimer(vault->timer_tc);
startv = nvtxs;
#pragma omp parallel for schedule(dynamic,1024) default(none) shared(nvtxs, xadj, adjncy, uxadj) private(vj, ei, ej) reduction(max: maxhmsize) reduction(min: startv)
for (vi=nvtxs-1; vi>=0; vi--) {
for (ei=xadj[vi+1]-1; adjncy[ei]>vi; ei--); 
uxadj[vi] = ei;
maxhmsize = gk_max(maxhmsize, (int32_t)(xadj[vi+1]-uxadj[vi]));
startv = (uxadj[vi] != xadj[vi] ? vi : startv);
for (ej=xadj[vi+1]-1; ei<ej; ei++, ej--) {
vj = adjncy[ei];
adjncy[ei] = adjncy[ej];
adjncy[ej] = vj;
}
}
for (l2=1; maxhmsize>(1<<l2); l2++);
maxhmsize = (1<<(l2+4))-1;
printf("& compatible maxhmsize: %"PRId32", startv: %d\n", maxhmsize, startv);
#pragma omp parallel default(none) shared(params, vault, nvtxs, xadj, adjncy, uxadj, maxhmsize, startv, stdout) reduction(+: ntriangles, nprobes)
{
int32_t vi, vj, vk, vl, nlocal;
ssize_t ei, eiend, eistart, ej, ejend, ejstart;
int32_t l, nc;
int32_t l2=1, hmsize=(1<<(l2+4))-1, *hmap;
#if defined(_OPENMP)
int mytid = omp_get_thread_num();
#else
int mytid = 0;
#endif
hmap = gk_i32smalloc(maxhmsize+1, 0, "hmap");
#pragma omp for schedule(dynamic,SBSIZE) nowait
for (vj=startv; vj<nvtxs-maxhmsize; vj++) {
if (xadj[vj+1]-uxadj[vj] == 1 || xadj[vj] == uxadj[vj])
continue;
if (xadj[vj+1]-uxadj[vj] > (1<<l2)) {
for (++l2; (xadj[vj+1]-uxadj[vj])>(1<<l2); l2++);
hmsize = (1<<(l2+4))-1;
}
for (nc=0, ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) {
vk = adjncy[ej];
for (l=(vk&hmsize); hmap[l]!=0; l=((l+1)&hmsize), nc++);
hmap[l] = vk;
}
nlocal = 0;
if (nc > 0) { 
for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
vi = adjncy[ej];
for (ei=uxadj[vi]; adjncy[ei]>vj; ei++) {
vk = adjncy[ei];
for (l=vk&hmsize; hmap[l]!=0 && hmap[l]!=vk; l=((l+1)&hmsize));
if (hmap[l] == vk) 
nlocal++;
}
nprobes += ei-uxadj[vi];
}
for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) {
vk = adjncy[ej];
for (l=(vk&hmsize); hmap[l]!=vk; l=((l+1)&hmsize));
hmap[l] = 0;
}
}
else { 
for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
vi = adjncy[ej];
#ifdef TC_VECOPT 
for (eiend=uxadj[vi]; adjncy[eiend]>vj; eiend++);
for (ei=uxadj[vi]; ei<eiend; ei++) 
#else
for (ei=uxadj[vi]; adjncy[ei]>vj; ei++) 
#endif
{
vk = adjncy[ei];
nlocal += (hmap[vk&hmsize] == vk);
}
nprobes += ei-uxadj[vi];
}
for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) 
hmap[adjncy[ej]&hmsize] = 0;
}
if (nlocal > 0)
ntriangles += nlocal;
}
hmap -= (nvtxs - maxhmsize);
#pragma omp for schedule(dynamic,DBSIZE) nowait
for (vj=nvtxs-1; vj>=nvtxs-maxhmsize; vj--) {
if (xadj[vj+1]-uxadj[vj] == 1 || xadj[vj] == uxadj[vj])
continue;
nlocal = 0;
if (xadj[vj+1]-uxadj[vj] == nvtxs-vj) { 
for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
vi = adjncy[ej];
for (ei=uxadj[vi]; adjncy[ei]>vj; ei++);
nlocal  += ei-uxadj[vi];
nprobes += ei-uxadj[vi];
}
}
else {
for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) 
hmap[adjncy[ej]] = 1;
for (ej=xadj[vj], ejend=uxadj[vj]; ej<ejend; ej++) {
vi = adjncy[ej];
#ifdef TC_VECOPT 
for (eiend=uxadj[vi]; adjncy[eiend]>vj; eiend++);
for (ei=uxadj[vi]; ei<eiend; ei++) 
#else
for (ei=uxadj[vi]; adjncy[ei]>vj; ei++) 
#endif
nlocal += hmap[adjncy[ei]];
nprobes += ei-uxadj[vi];
}
for (ej=uxadj[vj], ejend=xadj[vj+1]-1; ej<ejend; ej++) 
hmap[adjncy[ej]] = 0;
}
if (nlocal > 0)
ntriangles += nlocal;
}
hmap += (nvtxs - maxhmsize);
gk_free((void **)&hmap, LTERM);
}
gk_stopwctimer(vault->timer_tc);
gk_graph_Free(&graph);
gk_free((void **)&uxadj, LTERM);
vault->nprobes = nprobes;
return ntriangles;
}
