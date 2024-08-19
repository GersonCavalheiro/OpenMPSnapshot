

#include <GKlib.h>

#define OMPMINOPS       50000




gk_graph_t *gk_graph_Create() {
gk_graph_t *graph;

graph = (gk_graph_t *) gk_malloc(sizeof(gk_graph_t), "gk_graph_Create: graph");

gk_graph_Init(graph);

return graph;
}





void gk_graph_Init(gk_graph_t *graph) {
memset(graph, 0, sizeof(gk_graph_t));
graph->nvtxs = -1;
}





void gk_graph_Free(gk_graph_t **graph) {
if (*graph == NULL) {
return;
}
gk_graph_FreeContents(*graph);
gk_free((void **) graph, LTERM);
}





void gk_graph_FreeContents(gk_graph_t *graph) {
gk_free((void *) &graph->xadj, &graph->adjncy,
&graph->iadjwgt, &graph->fadjwgt,
&graph->ivwgts, &graph->fvwgts,
&graph->ivsizes, &graph->fvsizes,
&graph->vlabels,
LTERM);
}





gk_graph_t *gk_graph_Read(char *filename, int format, int isfewgts,
int isfvwgts, int isfvsizes) {
ssize_t i, k, l;
size_t nfields, nvtxs, nedges, fmt, ncon, lnlen;
int32_t ival;
float fval;
int readsizes = 0, readwgts = 0, readvals = 0, numbering = 0;
char *line = NULL, *head, *tail, fmtstr[256];
FILE *fpin = NULL;
gk_graph_t *graph = NULL;

if (!gk_fexists(filename)) {
gk_errexit(SIGERR, "File %s does not exist!\n", filename);
}

if (format == GK_GRAPH_FMT_METIS) {
fpin = gk_fopen(filename, "r", "gk_graph_Read: fpin");
do {
if (gk_getline(&line, &lnlen, fpin) <= 0) {
gk_errexit(SIGERR, "Premature end of input file: file:%s\n", filename);
}
} while (line[0] == '%');

fmt = ncon = 0;
nfields = sscanf(line, "%zu %zu %zu %zu", &nvtxs, &nedges, &fmt, &ncon);
if (nfields < 2) {
gk_errexit(SIGERR, "Header line must contain at least 2 integers (#vtxs and #edges).\n");
}

nedges *= 2;

if (fmt > 111) {
gk_errexit(SIGERR, "Cannot read this type of file format [fmt=%zu]!\n", fmt);
}

sprintf(fmtstr, "%03zu", fmt % 1000);
readsizes = (fmtstr[0] == '1');
readwgts = (fmtstr[1] == '1');
readvals = (fmtstr[2] == '1');
numbering = 1;
ncon = (ncon == 0 ? 1 : ncon);
} else {
gk_errexit(SIGERR, "Unrecognized format: %d\n", format);
}

graph = gk_graph_Create();

graph->nvtxs = nvtxs;

graph->xadj = gk_zmalloc(nvtxs + 1, "gk_graph_Read: xadj");
graph->adjncy = gk_i32malloc(nedges, "gk_graph_Read: adjncy");
if (readvals) {
if (isfewgts) {
graph->fadjwgt = gk_fmalloc(nedges, "gk_graph_Read: fadjwgt");
} else {
graph->iadjwgt = gk_i32malloc(nedges, "gk_graph_Read: iadjwgt");
}
}

if (readsizes) {
if (isfvsizes) {
graph->fvsizes = gk_fmalloc(nvtxs, "gk_graph_Read: fvsizes");
} else {
graph->ivsizes = gk_i32malloc(nvtxs, "gk_graph_Read: ivsizes");
}
}

if (readwgts) {
if (isfvwgts) {
graph->fvwgts = gk_fmalloc(nvtxs * ncon, "gk_graph_Read: fvwgts");
} else {
graph->ivwgts = gk_i32malloc(nvtxs * ncon, "gk_graph_Read: ivwgts");
}
}



numbering = (numbering ? -1 : 0);
for (graph->xadj[0] = 0, k = 0, i = 0; i < nvtxs; i++) {
do {
if (gk_getline(&line, &lnlen, fpin) == -1) {
gk_errexit(SIGERR, "Pregraphure end of input file: file while reading row %d\n", i);
}
} while (line[0] == '%');

head = line;
tail = NULL;


if (readsizes) {
if (isfvsizes) {
#ifdef __MSC__
graph->fvsizes[i] = (float)strtod(head, &tail);
#else
graph->fvsizes[i] = strtof(head, &tail);
#endif
if (tail == head) {
gk_errexit(SIGERR, "The line for vertex %zd does not have size information\n", i + 1);
}
if (graph->fvsizes[i] < 0) {
gk_errexit(SIGERR, "The size for vertex %zd must be >= 0\n", i + 1);
}
} else {
graph->ivsizes[i] = strtol(head, &tail, 0);
if (tail == head) {
gk_errexit(SIGERR, "The line for vertex %zd does not have size information\n", i + 1);
}
if (graph->ivsizes[i] < 0) {
gk_errexit(SIGERR, "The size for vertex %zd must be >= 0\n", i + 1);
}
}
head = tail;
}


if (readwgts) {
for (l = 0; l < ncon; l++) {
if (isfvwgts) {
#ifdef __MSC__
graph->fvwgts[i*ncon+l] = (float)strtod(head, &tail);
#else
graph->fvwgts[i * ncon + l] = strtof(head, &tail);
#endif
if (tail == head) {
gk_errexit(SIGERR, "The line for vertex %zd does not have enough weights "
"for the %d constraints.\n", i + 1, ncon);
}
if (graph->fvwgts[i * ncon + l] < 0) {
gk_errexit(SIGERR, "The weight vertex %zd and constraint %zd must be >= 0\n", i + 1, l);
}
} else {
graph->ivwgts[i * ncon + l] = strtol(head, &tail, 0);
if (tail == head) {
gk_errexit(SIGERR, "The line for vertex %zd does not have enough weights "
"for the %d constraints.\n", i + 1, ncon);
}
if (graph->ivwgts[i * ncon + l] < 0) {
gk_errexit(SIGERR, "The weight vertex %zd and constraint %zd must be >= 0\n", i + 1, l);
}
}
head = tail;
}
}



while (1) {
ival = (int) strtol(head, &tail, 0);
if (tail == head) {
break;
}
head = tail;

if ((graph->adjncy[k] = ival + numbering) < 0) {
gk_errexit(SIGERR, "Error: Invalid column number %d at row %zd.\n", ival, i);
}

if (readvals) {
if (isfewgts) {
#ifdef __MSC__
fval = (float)strtod(head, &tail);
#else
fval = strtof(head, &tail);
#endif
if (tail == head) {
gk_errexit(SIGERR, "Value could not be found for edge! Vertex:%zd, NNZ:%zd\n", i, k);
}

graph->fadjwgt[k] = fval;
} else {
ival = strtol(head, &tail, 0);
if (tail == head) {
gk_errexit(SIGERR, "Value could not be found for edge! Vertex:%zd, NNZ:%zd\n", i, k);
}

graph->iadjwgt[k] = ival;
}
head = tail;
}
k++;
}
graph->xadj[i + 1] = k;
}

if (k != nedges) {
gk_errexit(SIGERR, "gk_graph_Read: Something wrong with the number of edges in "
"the input file. nedges=%zd, Actualnedges=%zd.\n", nedges, k);
}

gk_fclose(fpin);

gk_free((void **) &line, LTERM);

return graph;
}





void gk_graph_Write(gk_graph_t *graph, char *filename, int format) {
ssize_t i, j;
int hasvwgts, hasvsizes, hasewgts;
FILE *fpout;

if (format != GK_GRAPH_FMT_METIS) {
gk_errexit(SIGERR, "Unknown file format. %d\n", format);
}

if (filename) {
fpout = gk_fopen(filename, "w", "gk_graph_Write: fpout");
} else {
fpout = stdout;
}

hasewgts = (graph->iadjwgt || graph->fadjwgt);
hasvwgts = (graph->ivwgts || graph->fvwgts);
hasvsizes = (graph->ivsizes || graph->fvsizes);


fprintf(fpout, "%d %zd", graph->nvtxs, graph->xadj[graph->nvtxs] / 2);
if (hasvwgts || hasvsizes || hasewgts) {
fprintf(fpout, " %d%d%d", hasvsizes, hasvwgts, hasewgts);
}
fprintf(fpout, "\n");

for (i = 0; i < graph->nvtxs; i++) {
if (hasvsizes) {
if (graph->ivsizes) {
fprintf(fpout, " %d", graph->ivsizes[i]);
} else {
fprintf(fpout, " %f", graph->fvsizes[i]);
}
}

if (hasvwgts) {
if (graph->ivwgts) {
fprintf(fpout, " %d", graph->ivwgts[i]);
} else {
fprintf(fpout, " %f", graph->fvwgts[i]);
}
}

for (j = graph->xadj[i]; j < graph->xadj[i + 1]; j++) {
fprintf(fpout, " %d", graph->adjncy[j] + 1);
if (hasewgts) {
if (graph->iadjwgt) {
fprintf(fpout, " %d", graph->iadjwgt[j]);
} else {
fprintf(fpout, " %f", graph->fadjwgt[j]);
}
}
}
fprintf(fpout, "\n");
}
if (filename) {
gk_fclose(fpout);
}
}





gk_graph_t *gk_graph_Dup(gk_graph_t *graph) {
gk_graph_t *ngraph;

ngraph = gk_graph_Create();

ngraph->nvtxs = graph->nvtxs;


if (graph->xadj) {
ngraph->xadj = gk_zcopy(graph->nvtxs + 1, graph->xadj,
gk_zmalloc(graph->nvtxs + 1, "gk_graph_Dup: xadj"));
}
if (graph->ivwgts) {
ngraph->ivwgts = gk_i32copy(graph->nvtxs, graph->ivwgts,
gk_i32malloc(graph->nvtxs, "gk_graph_Dup: ivwgts"));
}
if (graph->ivsizes) {
ngraph->ivsizes = gk_i32copy(graph->nvtxs, graph->ivsizes,
gk_i32malloc(graph->nvtxs, "gk_graph_Dup: ivsizes"));
}
if (graph->vlabels) {
ngraph->vlabels = gk_i32copy(graph->nvtxs, graph->vlabels,
gk_i32malloc(graph->nvtxs, "gk_graph_Dup: ivlabels"));
}
if (graph->fvwgts) {
ngraph->fvwgts = gk_fcopy(graph->nvtxs, graph->fvwgts,
gk_fmalloc(graph->nvtxs, "gk_graph_Dup: fvwgts"));
}
if (graph->fvsizes) {
ngraph->fvsizes = gk_fcopy(graph->nvtxs, graph->fvsizes,
gk_fmalloc(graph->nvtxs, "gk_graph_Dup: fvsizes"));
}

if (graph->adjncy) {
ngraph->adjncy = gk_i32copy(graph->xadj[graph->nvtxs], graph->adjncy,
gk_i32malloc(graph->xadj[graph->nvtxs], "gk_graph_Dup: adjncy"));
}
if (graph->iadjwgt) {
ngraph->iadjwgt = gk_i32copy(graph->xadj[graph->nvtxs], graph->iadjwgt,
gk_i32malloc(graph->xadj[graph->nvtxs], "gk_graph_Dup: iadjwgt"));
}
if (graph->fadjwgt) {
ngraph->fadjwgt = gk_fcopy(graph->xadj[graph->nvtxs], graph->fadjwgt,
gk_fmalloc(graph->xadj[graph->nvtxs], "gk_graph_Dup: fadjwgt"));
}

return ngraph;
}





gk_graph_t *gk_graph_ExtractSubgraph(gk_graph_t *graph, int vstart, int nvtxs) {
ssize_t i;
gk_graph_t *ngraph;

if (vstart + nvtxs > graph->nvtxs) {
return NULL;
}

ngraph = gk_graph_Create();

ngraph->nvtxs = nvtxs;


if (graph->xadj) {
ngraph->xadj = gk_zcopy(nvtxs + 1, graph->xadj + vstart,
gk_zmalloc(nvtxs + 1, "gk_graph_ExtractSubgraph: xadj"));
}
for (i = nvtxs; i >= 0; i--) {
ngraph->xadj[i] -= ngraph->xadj[0];
}
ASSERT(ngraph->xadj[0] == 0);

if (graph->ivwgts) {
ngraph->ivwgts = gk_i32copy(nvtxs, graph->ivwgts + vstart,
gk_i32malloc(nvtxs, "gk_graph_ExtractSubgraph: ivwgts"));
}
if (graph->ivsizes) {
ngraph->ivsizes = gk_i32copy(nvtxs, graph->ivsizes + vstart,
gk_i32malloc(nvtxs, "gk_graph_ExtractSubgraph: ivsizes"));
}
if (graph->vlabels) {
ngraph->vlabels = gk_i32copy(nvtxs, graph->vlabels + vstart,
gk_i32malloc(nvtxs, "gk_graph_ExtractSubgraph: vlabels"));
}

if (graph->fvwgts) {
ngraph->fvwgts = gk_fcopy(nvtxs, graph->fvwgts + vstart,
gk_fmalloc(nvtxs, "gk_graph_ExtractSubgraph: fvwgts"));
}
if (graph->fvsizes) {
ngraph->fvsizes = gk_fcopy(nvtxs, graph->fvsizes + vstart,
gk_fmalloc(nvtxs, "gk_graph_ExtractSubgraph: fvsizes"));
}

ASSERT(ngraph->xadj[nvtxs] == graph->xadj[vstart + nvtxs] - graph->xadj[vstart]);
if (graph->adjncy) {
ngraph->adjncy = gk_i32copy(graph->xadj[vstart + nvtxs] - graph->xadj[vstart],
graph->adjncy + graph->xadj[vstart],
gk_i32malloc(graph->xadj[vstart + nvtxs] - graph->xadj[vstart],
"gk_graph_ExtractSubgraph: adjncy"));
}
if (graph->iadjwgt) {
ngraph->iadjwgt = gk_i32copy(graph->xadj[vstart + nvtxs] - graph->xadj[vstart],
graph->iadjwgt + graph->xadj[vstart],
gk_i32malloc(graph->xadj[vstart + nvtxs] - graph->xadj[vstart],
"gk_graph_ExtractSubgraph: iadjwgt"));
}
if (graph->fadjwgt) {
ngraph->fadjwgt = gk_fcopy(graph->xadj[vstart + nvtxs] - graph->xadj[vstart],
graph->fadjwgt + graph->xadj[vstart],
gk_fmalloc(graph->xadj[vstart + nvtxs] - graph->xadj[vstart],
"gk_graph_ExtractSubgraph: fadjwgt"));
}

return ngraph;
}





gk_graph_t *gk_graph_Reorder(gk_graph_t *graph, int32_t *perm, int32_t *iperm) {
ssize_t j, jj, *xadj;
int i, k, u, v, nvtxs;
int freeperm = 0, freeiperm = 0;
int32_t *adjncy;
gk_graph_t *ngraph;

if (perm == NULL && iperm == NULL) {
return NULL;
}

ngraph = gk_graph_Create();

ngraph->nvtxs = nvtxs = graph->nvtxs;
xadj = graph->xadj;
adjncy = graph->adjncy;


if (graph->xadj) {
ngraph->xadj = gk_zmalloc(nvtxs + 1, "gk_graph_Reorder: xadj");
}

if (graph->ivwgts) {
ngraph->ivwgts = gk_i32malloc(nvtxs, "gk_graph_Reorder: ivwgts");
}

if (graph->ivsizes) {
ngraph->ivsizes = gk_i32malloc(nvtxs, "gk_graph_Reorder: ivsizes");
}

if (graph->vlabels) {
ngraph->vlabels = gk_i32malloc(nvtxs, "gk_graph_Reorder: ivlabels");
}

if (graph->fvwgts) {
ngraph->fvwgts = gk_fmalloc(nvtxs, "gk_graph_Reorder: fvwgts");
}

if (graph->fvsizes) {
ngraph->fvsizes = gk_fmalloc(nvtxs, "gk_graph_Reorder: fvsizes");
}

if (graph->adjncy) {
ngraph->adjncy = gk_i32malloc(graph->xadj[nvtxs], "gk_graph_Reorder: adjncy");
}

if (graph->iadjwgt) {
ngraph->iadjwgt = gk_i32malloc(graph->xadj[nvtxs], "gk_graph_Reorder: iadjwgt");
}

if (graph->fadjwgt) {
ngraph->fadjwgt = gk_fmalloc(graph->xadj[nvtxs], "gk_graph_Reorder: fadjwgt");
}



if (perm == NULL) {
freeperm = 1;
perm = gk_i32malloc(nvtxs, "gk_graph_Reorder: perm");
for (i = 0; i < nvtxs; i++) {
perm[iperm[i]] = i;
}
}
if (iperm == NULL) {
freeiperm = 1;
iperm = gk_i32malloc(nvtxs, "gk_graph_Reorder: iperm");
for (i = 0; i < nvtxs; i++) {
iperm[perm[i]] = i;
}
}


ngraph->xadj[0] = jj = 0;
for (v = 0; v < nvtxs; v++) {
u = iperm[v];
for (j = xadj[u]; j < xadj[u + 1]; j++, jj++) {
ngraph->adjncy[jj] = perm[adjncy[j]];
if (graph->iadjwgt) {
ngraph->iadjwgt[jj] = graph->iadjwgt[j];
}
if (graph->fadjwgt) {
ngraph->fadjwgt[jj] = graph->fadjwgt[j];
}
}
if (graph->ivwgts) {
ngraph->ivwgts[v] = graph->ivwgts[u];
}
if (graph->fvwgts) {
ngraph->fvwgts[v] = graph->fvwgts[u];
}
if (graph->ivsizes) {
ngraph->ivsizes[v] = graph->ivsizes[u];
}
if (graph->fvsizes) {
ngraph->fvsizes[v] = graph->fvsizes[u];
}
if (graph->vlabels) {
ngraph->vlabels[v] = graph->vlabels[u];
}

ngraph->xadj[v + 1] = jj;
}



if (freeperm) {
gk_free((void **) &perm, LTERM);
}
if (freeiperm) {
gk_free((void **) &iperm, LTERM);
}

return ngraph;
}





int gk_graph_FindComponents(gk_graph_t *graph, int32_t *cptr, int32_t *cind) {
ssize_t i, ii, j, jj, k, nvtxs, first, last, ntodo, ncmps;
ssize_t *xadj;
int32_t *adjncy, *pos, *todo;
int32_t mustfree_ccsr = 0, mustfree_where = 0;

nvtxs = graph->nvtxs;
xadj = graph->xadj;
adjncy = graph->adjncy;


if (cptr == NULL) {
cptr = gk_i32malloc(nvtxs + 1, "gk_graph_FindComponents: cptr");
cind = gk_i32malloc(nvtxs, "gk_graph_FindComponents: cind");
mustfree_ccsr = 1;
}


todo = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_graph_FindComponents: todo"));


pos = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_graph_FindComponents: pos"));



ncmps = -1;
ntodo = nvtxs;     
first = last = 0;  
while (ntodo > 0) {
if (first == last) { 
cptr[++ncmps] = first;  

ASSERT(pos[todo[0]] != -1);
i = todo[0];

cind[last++] = i;
pos[i] = -1;
}

i = cind[first++];  


k = pos[i];
j = todo[k] = todo[--ntodo];
pos[j] = k;

for (j = xadj[i]; j < xadj[i + 1]; j++) {
k = adjncy[j];
if (pos[k] != -1) {
cind[last++] = k;
pos[k] = -1;
}
}
}
cptr[++ncmps] = first;

if (mustfree_ccsr) {
gk_free((void **) &cptr, &cind, LTERM);
}

gk_free((void **) &pos, &todo, LTERM);

return (int) ncmps;
}





void gk_graph_ComputeBFSOrdering(gk_graph_t *graph, int v, int32_t **r_perm,
int32_t **r_iperm) {
ssize_t j, *xadj;
int i, k, nvtxs, first, last;
int32_t *adjncy, *cot, *pos;

if (graph->nvtxs <= 0) {
return;
}

nvtxs = graph->nvtxs;
xadj = graph->xadj;
adjncy = graph->adjncy;


pos = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_graph_ComputeBFSOrdering: pos"));


cot = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_graph_ComputeBFSOrdering: cot"));



pos[0] = cot[0] = v;
pos[v] = cot[v] = 0;


first = last = 0;
while (first < nvtxs) {
if (first == last) { 
k = cot[last];
ASSERT(pos[k] != -1);
pos[k] = -1; 
last++;
}

i = cot[first++];  
for (j = xadj[i]; j < xadj[i + 1]; j++) {
k = adjncy[j];

if (pos[k] != -1) {

cot[pos[k]] = cot[last]; 
pos[cot[last]] = pos[k];    

cot[last++] = k;  
pos[k] = -1; 
}
}
}


if (r_perm != NULL) {

for (i = 0; i < nvtxs; i++) {
pos[cot[i]] = i;
}

*r_perm = pos;
pos = NULL;
}

if (r_iperm != NULL) {
*r_iperm = cot;
cot = NULL;
}



gk_free((void **) &pos, &cot, LTERM);

}





void gk_graph_ComputeBestFOrdering0(gk_graph_t *graph, int v, int type,
int32_t **r_perm, int32_t **r_iperm) {
ssize_t j, jj, *xadj;
int i, k, u, nvtxs;
int32_t *adjncy, *perm, *degrees, *minIDs, *open;
gk_i32pq_t *queue;

if (graph->nvtxs <= 0) {
return;
}

nvtxs = graph->nvtxs;
xadj = graph->xadj;
adjncy = graph->adjncy;


degrees = gk_i32smalloc(nvtxs, 0, "gk_graph_ComputeBestFOrdering: degrees");


minIDs = gk_i32smalloc(nvtxs, nvtxs + 1, "gk_graph_ComputeBestFOrdering: minIDs");


open = gk_i32malloc(nvtxs, "gk_graph_ComputeBestFOrdering: open");


perm = gk_i32smalloc(nvtxs, -1, "gk_graph_ComputeBestFOrdering: perm");


queue = gk_i32pqCreate(nvtxs);
for (i = 0; i < nvtxs; i++) {
gk_i32pqInsert(queue, i, 0);
}
gk_i32pqUpdate(queue, v, 1);

open[0] = v;


for (i = 0; i < nvtxs; i++) {
if ((v = gk_i32pqGetTop(queue)) == -1) {
gk_errexit(SIGERR, "The priority queue got empty ahead of time [i=%d].\n", i);
}
if (perm[v] != -1) {
gk_errexit(SIGERR, "The perm[%d] has already been set.\n", v);
}
perm[v] = i;

for (j = xadj[v]; j < xadj[v + 1]; j++) {
u = adjncy[j];
if (perm[u] == -1) {
degrees[u]++;
minIDs[u] = (i < minIDs[u] ? i : minIDs[u]);

switch (type) {
case 1: 
gk_i32pqUpdate(queue, u, 1);
break;
case 2: 
gk_i32pqUpdate(queue, u, degrees[u]);
break;
case 3: 
for (k = 0, jj = xadj[u]; jj < xadj[u + 1]; jj++) {
if (perm[adjncy[jj]] != -1) {
k += perm[adjncy[jj]];
}
}
gk_i32pqUpdate(queue, u, k);
break;
case 4: 
for (k = 0, jj = xadj[u]; jj < xadj[u + 1]; jj++) {
if (perm[adjncy[jj]] != -1) {
k += (i - perm[adjncy[jj]]);
}
}
gk_i32pqUpdate(queue, u, k);
break;
default:;
}
}
}
}



if (r_perm != NULL) {
*r_perm = perm;
perm = NULL;
}

if (r_iperm != NULL) {

for (i = 0; i < nvtxs; i++) {
degrees[perm[i]] = i;
}

*r_iperm = degrees;
degrees = NULL;
}




gk_i32pqDestroy(queue);
gk_free((void **) &perm, &degrees, &minIDs, &open, LTERM);

}





void gk_graph_ComputeBestFOrdering(gk_graph_t *graph, int v, int type,
int32_t **r_perm, int32_t **r_iperm) {
ssize_t j, jj, *xadj;
int i, k, u, nvtxs, nopen, ntodo;
int32_t *adjncy, *perm, *degrees, *wdegrees, *sod, *level, *ot, *pos;
gk_i32pq_t *queue;

if (graph->nvtxs <= 0) {
return;
}

nvtxs = graph->nvtxs;
xadj = graph->xadj;
adjncy = graph->adjncy;


degrees = gk_i32smalloc(nvtxs, 0, "gk_graph_ComputeBestFOrdering: degrees");


wdegrees = gk_i32smalloc(nvtxs, 0, "gk_graph_ComputeBestFOrdering: wdegrees");


sod = gk_i32smalloc(nvtxs, 0, "gk_graph_ComputeBestFOrdering: sod");


level = gk_i32smalloc(nvtxs, 0, "gk_graph_ComputeBestFOrdering: level");


ot = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_graph_FindComponents: ot"));


pos = gk_i32incset(nvtxs, 0, gk_i32malloc(nvtxs, "gk_graph_FindComponents: pos"));


perm = gk_i32smalloc(nvtxs, -1, "gk_graph_ComputeBestFOrdering: perm");


queue = gk_i32pqCreate(nvtxs);
gk_i32pqInsert(queue, v, 1);


pos[0] = ot[0] = v;
pos[v] = ot[v] = 0;
nopen = 1;
ntodo = nvtxs;


for (i = 0; i < nvtxs; i++) {
if (nopen == 0) { 
gk_i32pqInsert(queue, ot[0], 1);
nopen++;
}

if ((v = gk_i32pqGetTop(queue)) == -1) {
gk_errexit(SIGERR, "The priority queue got empty ahead of time [i=%d].\n", i);
}

if (perm[v] != -1) {
gk_errexit(SIGERR, "The perm[%d] has already been set.\n", v);
}
perm[v] = i;

if (ot[pos[v]] != v) {
gk_errexit(SIGERR, "Something went wrong [ot[pos[%d]]!=%d.\n", v, v);
}
if (pos[v] >= nopen) {
gk_errexit(SIGERR, "The position of v is not in open list. pos[%d]=%d is >=%d.\n", v, pos[v], nopen);
}


ot[pos[v]] = ot[nopen - 1];
pos[ot[nopen - 1]] = pos[v];
if (ntodo > nopen) {
ot[nopen - 1] = ot[ntodo - 1];
pos[ot[ntodo - 1]] = nopen - 1;
}
nopen--;
ntodo--;

for (j = xadj[v]; j < xadj[v + 1]; j++) {
u = adjncy[j];
if (perm[u] == -1) {

if (degrees[u] == 0) {
ot[pos[u]] = ot[nopen];
pos[ot[nopen]] = pos[u];
ot[nopen] = u;
pos[u] = nopen;
nopen++;

level[u] = level[v] + 1;
gk_i32pqInsert(queue, u, 0);
}



degrees[u]++;


switch (type) {
case 1: 
gk_i32pqUpdate(queue, u, 1000 * (i + 1) + degrees[u]);
break;

case 2: 
gk_i32pqUpdate(queue, u, degrees[u]);
break;

case 3: 
wdegrees[u] += i;
gk_i32pqUpdate(queue, u, wdegrees[u]);
break;

case 4: 

;
break;

case 5: 
gk_i32pqUpdate(queue, u, -(1000 * level[u] - degrees[u]));
break;

case 6: 
gk_i32pqUpdate(queue, u, (i + 1) * degrees[u]);
break;

default:;
}
}
}

if (type == 4) { 
for (j = 0; j < nopen; j++) {
u = ot[j];
if (perm[u] != -1) {
gk_errexit(SIGERR,
"For i=%d, the open list contains a closed vertex: ot[%zd]=%d, perm[%d]=%d.\n",
i,
j,
u,
u,
perm[u]);
}
sod[u] += degrees[u];
if (i < 1000 || i % 25 == 0) {
gk_i32pqUpdate(queue, u, sod[u]);
}
}
}



}



if (r_perm != NULL) {
*r_perm = perm;
perm = NULL;
}

if (r_iperm != NULL) {

for (i = 0; i < nvtxs; i++) {
degrees[perm[i]] = i;
}

*r_iperm = degrees;
degrees = NULL;
}




gk_i32pqDestroy(queue);
gk_free((void **) &perm, &degrees, &wdegrees, &sod, &ot, &pos, &level, LTERM);

}





void gk_graph_SingleSourceShortestPaths(gk_graph_t *graph, int v, void **r_sps) {
ssize_t *xadj;
int i, u, nvtxs;
int32_t *adjncy, *inqueue;

if (graph->nvtxs <= 0) {
return;
}

nvtxs = graph->nvtxs;
xadj = graph->xadj;
adjncy = graph->adjncy;

inqueue = gk_i32smalloc(nvtxs, 0, "gk_graph_SingleSourceShortestPaths: inqueue");


if (graph->iadjwgt != NULL) {
gk_i32pq_t *queue;
int32_t *adjwgt;
int32_t *sps;

adjwgt = graph->iadjwgt;

queue = gk_i32pqCreate(nvtxs);
gk_i32pqInsert(queue, v, 0);
inqueue[v] = 1;

sps = gk_i32smalloc(nvtxs, -1, "gk_graph_SingleSourceShortestPaths: sps");
sps[v] = 0;


while ((v = gk_i32pqGetTop(queue)) != -1) {
inqueue[v] = 2;


for (i = xadj[v]; i < xadj[v + 1]; i++) {
u = adjncy[i];
if (inqueue[u] == 2) {
continue;
}

if (sps[u] < 0 || sps[v] + adjwgt[i] < sps[u]) {
sps[u] = sps[v] + adjwgt[i];

if (inqueue[u]) {
gk_i32pqUpdate(queue, u, -sps[u]);
} else {
gk_i32pqInsert(queue, u, -sps[u]);
inqueue[u] = 1;
}
}
}
}

*r_sps = (void *) sps;

gk_i32pqDestroy(queue);
} else {
gk_fpq_t *queue;
float *adjwgt;
float *sps;

adjwgt = graph->fadjwgt;

queue = gk_fpqCreate(nvtxs);
gk_fpqInsert(queue, v, 0);
inqueue[v] = 1;

sps = gk_fsmalloc(nvtxs, -1, "gk_graph_SingleSourceShortestPaths: sps");
sps[v] = 0;


while ((v = gk_fpqGetTop(queue)) != -1) {
inqueue[v] = 2;


for (i = xadj[v]; i < xadj[v + 1]; i++) {
u = adjncy[i];
if (inqueue[u] == 2) {
continue;
}

if (sps[u] < 0 || sps[v] + adjwgt[i] < sps[u]) {
sps[u] = sps[v] + adjwgt[i];

if (inqueue[u]) {
gk_fpqUpdate(queue, u, -sps[u]);
} else {
gk_fpqInsert(queue, u, -sps[u]);
inqueue[u] = 1;
}
}
}
}

*r_sps = (void *) sps;

gk_fpqDestroy(queue);
}

gk_free((void **) &inqueue, LTERM);

}

#ifdef XXX




void gk_graph_SortAdjacencies(gk_graph_t *graph)
{
int n, nn=0;
ssize_t *ptr;
int *ind;
float *val;

switch (what) {
case GK_CSR_ROW:
if (!graph->rowptr)
gk_errexit(SIGERR, "Row-based view of the graphrix does not exists.\n");

n   = graph->nrows;
ptr = graph->rowptr;
ind = graph->rowind;
val = graph->rowval;
break;

case GK_CSR_COL:
if (!graph->colptr)
gk_errexit(SIGERR, "Column-based view of the graphrix does not exists.\n");

n   = graph->ncols;
ptr = graph->colptr;
ind = graph->colind;
val = graph->colval;
break;

default:
gk_errexit(SIGERR, "Invalid index type of %d.\n", what);
return;
}

#pragma omp parallel if (n > 100)
{
ssize_t i, j, k;
gk_ikv_t *cand;
float *tval;

#pragma omp single
for (i=0; i<n; i++)
nn = gk_max(nn, ptr[i+1]-ptr[i]);

cand = gk_ikvmalloc(nn, "gk_graph_SortIndices: cand");
tval = gk_fmalloc(nn, "gk_graph_SortIndices: tval");

#pragma omp for schedule(static)
for (i=0; i<n; i++) {
for (k=0, j=ptr[i]; j<ptr[i+1]; j++) {
if (j > ptr[i] && ind[j] < ind[j-1])
k = 1; 
cand[j-ptr[i]].val = j-ptr[i];
cand[j-ptr[i]].key = ind[j];
tval[j-ptr[i]]     = val[j];
}
if (k) {
gk_ikvsorti(ptr[i+1]-ptr[i], cand);
for (j=ptr[i]; j<ptr[i+1]; j++) {
ind[j] = cand[j-ptr[i]].key;
val[j] = tval[cand[j-ptr[i]].val];
}
}
}

gk_free((void **)&cand, &tval, LTERM);
}

}





gk_graph_t *gk_graph_ExtractRows(gk_graph_t *graph, int nrows, int *rind)
{
ssize_t i, ii, j, nnz;
gk_graph_t *ngraph;

ngraph = gk_graph_Create();

ngraph->nrows = nrows;
ngraph->ncols = graph->ncols;

for (nnz=0, i=0; i<nrows; i++)
nnz += graph->rowptr[rind[i]+1]-graph->rowptr[rind[i]];

ngraph->rowptr = gk_zmalloc(ngraph->nrows+1, "gk_graph_ExtractPartition: rowptr");
ngraph->rowind = gk_imalloc(nnz, "gk_graph_ExtractPartition: rowind");
ngraph->rowval = gk_fmalloc(nnz, "gk_graph_ExtractPartition: rowval");

ngraph->rowptr[0] = 0;
for (nnz=0, j=0, ii=0; ii<nrows; ii++) {
i = rind[ii];
gk_icopy(graph->rowptr[i+1]-graph->rowptr[i], graph->rowind+graph->rowptr[i], ngraph->rowind+nnz);
gk_fcopy(graph->rowptr[i+1]-graph->rowptr[i], graph->rowval+graph->rowptr[i], ngraph->rowval+nnz);
nnz += graph->rowptr[i+1]-graph->rowptr[i];
ngraph->rowptr[++j] = nnz;
}
ASSERT(j == ngraph->nrows);

return ngraph;
}





gk_graph_t *gk_graph_ExtractPartition(gk_graph_t *graph, int *part, int pid)
{
ssize_t i, j, nnz;
gk_graph_t *ngraph;

ngraph = gk_graph_Create();

ngraph->nrows = 0;
ngraph->ncols = graph->ncols;

for (nnz=0, i=0; i<graph->nrows; i++) {
if (part[i] == pid) {
ngraph->nrows++;
nnz += graph->rowptr[i+1]-graph->rowptr[i];
}
}

ngraph->rowptr = gk_zmalloc(ngraph->nrows+1, "gk_graph_ExtractPartition: rowptr");
ngraph->rowind = gk_imalloc(nnz, "gk_graph_ExtractPartition: rowind");
ngraph->rowval = gk_fmalloc(nnz, "gk_graph_ExtractPartition: rowval");

ngraph->rowptr[0] = 0;
for (nnz=0, j=0, i=0; i<graph->nrows; i++) {
if (part[i] == pid) {
gk_icopy(graph->rowptr[i+1]-graph->rowptr[i], graph->rowind+graph->rowptr[i], ngraph->rowind+nnz);
gk_fcopy(graph->rowptr[i+1]-graph->rowptr[i], graph->rowval+graph->rowptr[i], ngraph->rowval+nnz);
nnz += graph->rowptr[i+1]-graph->rowptr[i];
ngraph->rowptr[++j] = nnz;
}
}
ASSERT(j == ngraph->nrows);

return ngraph;
}





gk_graph_t **gk_graph_Split(gk_graph_t *graph, int *color)
{
ssize_t i, j;
int nrows, ncolors;
ssize_t *rowptr;
int *rowind;
float *rowval;
gk_graph_t **sgraphs;

nrows  = graph->nrows;
rowptr = graph->rowptr;
rowind = graph->rowind;
rowval = graph->rowval;

ncolors = gk_imax(rowptr[nrows], color)+1;

sgraphs = (gk_graph_t **)gk_malloc(sizeof(gk_graph_t *)*ncolors, "gk_graph_Split: sgraphs");
for (i=0; i<ncolors; i++) {
sgraphs[i] = gk_graph_Create();
sgraphs[i]->nrows  = graph->nrows;
sgraphs[i]->ncols  = graph->ncols;
sgraphs[i]->rowptr = gk_zsmalloc(nrows+1, 0, "gk_graph_Split: sgraphs[i]->rowptr");
}

for (i=0; i<nrows; i++) {
for (j=rowptr[i]; j<rowptr[i+1]; j++)
sgraphs[color[j]]->rowptr[i]++;
}
for (i=0; i<ncolors; i++)
MAKECSR(j, nrows, sgraphs[i]->rowptr);

for (i=0; i<ncolors; i++) {
sgraphs[i]->rowind = gk_imalloc(sgraphs[i]->rowptr[nrows], "gk_graph_Split: sgraphs[i]->rowind");
sgraphs[i]->rowval = gk_fmalloc(sgraphs[i]->rowptr[nrows], "gk_graph_Split: sgraphs[i]->rowval");
}

for (i=0; i<nrows; i++) {
for (j=rowptr[i]; j<rowptr[i+1]; j++) {
sgraphs[color[j]]->rowind[sgraphs[color[j]]->rowptr[i]] = rowind[j];
sgraphs[color[j]]->rowval[sgraphs[color[j]]->rowptr[i]] = rowval[j];
sgraphs[color[j]]->rowptr[i]++;
}
}

for (i=0; i<ncolors; i++)
SHIFTCSR(j, nrows, sgraphs[i]->rowptr);

return sgraphs;
}





gk_graph_t *gk_graph_Prune(gk_graph_t *graph, int what, int minf, int maxf)
{
ssize_t i, j, nnz;
int nrows, ncols;
ssize_t *rowptr, *nrowptr;
int *rowind, *nrowind, *collen;
float *rowval, *nrowval;
gk_graph_t *ngraph;

ngraph = gk_graph_Create();

nrows = ngraph->nrows = graph->nrows;
ncols = ngraph->ncols = graph->ncols;

rowptr = graph->rowptr;
rowind = graph->rowind;
rowval = graph->rowval;

nrowptr = ngraph->rowptr = gk_zmalloc(nrows+1, "gk_graph_Prune: nrowptr");
nrowind = ngraph->rowind = gk_imalloc(rowptr[nrows], "gk_graph_Prune: nrowind");
nrowval = ngraph->rowval = gk_fmalloc(rowptr[nrows], "gk_graph_Prune: nrowval");


switch (what) {
case GK_CSR_COL:
collen = gk_ismalloc(ncols, 0, "gk_graph_Prune: collen");

for (i=0; i<nrows; i++) {
for (j=rowptr[i]; j<rowptr[i+1]; j++) {
ASSERT(rowind[j] < ncols);
collen[rowind[j]]++;
}
}
for (i=0; i<ncols; i++)
collen[i] = (collen[i] >= minf && collen[i] <= maxf ? 1 : 0);

nrowptr[0] = 0;
for (nnz=0, i=0; i<nrows; i++) {
for (j=rowptr[i]; j<rowptr[i+1]; j++) {
if (collen[rowind[j]]) {
nrowind[nnz] = rowind[j];
nrowval[nnz] = rowval[j];
nnz++;
}
}
nrowptr[i+1] = nnz;
}
gk_free((void **)&collen, LTERM);
break;

case GK_CSR_ROW:
nrowptr[0] = 0;
for (nnz=0, i=0; i<nrows; i++) {
if (rowptr[i+1]-rowptr[i] >= minf && rowptr[i+1]-rowptr[i] <= maxf) {
for (j=rowptr[i]; j<rowptr[i+1]; j++, nnz++) {
nrowind[nnz] = rowind[j];
nrowval[nnz] = rowval[j];
}
}
nrowptr[i+1] = nnz;
}
break;

default:
gk_graph_Free(&ngraph);
gk_errexit(SIGERR, "Unknown prunning type of %d\n", what);
return NULL;
}

return ngraph;
}






void gk_graph_Normalize(gk_graph_t *graph, int what, int norm)
{
ssize_t i, j;
int n;
ssize_t *ptr;
float *val, sum;

if (what&GK_CSR_ROW && graph->rowval) {
n   = graph->nrows;
ptr = graph->rowptr;
val = graph->rowval;

#pragma omp parallel if (ptr[n] > OMPMINOPS)
{
#pragma omp for private(j,sum) schedule(static)
for (i=0; i<n; i++) {
for (sum=0.0, j=ptr[i]; j<ptr[i+1]; j++){
if (norm == 2)
sum += val[j]*val[j];
else if (norm == 1)
sum += val[j]; 
}
if (sum > 0) {
if (norm == 2)
sum=1.0/sqrt(sum);
else if (norm == 1)
sum=1.0/sum;
for (j=ptr[i]; j<ptr[i+1]; j++)
val[j] *= sum;

}
}
}
}

if (what&GK_CSR_COL && graph->colval) {
n   = graph->ncols;
ptr = graph->colptr;
val = graph->colval;

#pragma omp parallel if (ptr[n] > OMPMINOPS)
{
#pragma omp for private(j,sum) schedule(static)
for (i=0; i<n; i++) {
for (sum=0.0, j=ptr[i]; j<ptr[i+1]; j++)
if (norm == 2)
sum += val[j]*val[j];
else if (norm == 1)
sum += val[j];
if (sum > 0) {
if (norm == 2)
sum=1.0/sqrt(sum);
else if (norm == 1)
sum=1.0/sum;
for (j=ptr[i]; j<ptr[i+1]; j++)
val[j] *= sum;
}
}
}
}
}


#endif
