#pragma once



void print_comp_mins(ComponentSpace cs, Graph graph, DistT *minwtcomponent, unsigned *goaheadnodeofcomponent, unsigned *partners, bool *pin)
{
DistT *cminwt;
unsigned *cgah, *cpart;
unsigned *ele2comp;
bool *cpin;

ele2comp = (unsigned *) calloc(cs.nelements, sizeof(unsigned));
cgah = (unsigned *) calloc(cs.nelements, sizeof(unsigned));
cpart = (unsigned *) calloc(cs.nelements, sizeof(unsigned));
cminwt = (DistT *) calloc(cs.nelements, sizeof(unsigned));
cpin = (bool *) calloc(cs.nelements, sizeof(bool));


assert(cudaMemcpy(ele2comp, cs.ele2comp, cs.nelements * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
assert(cudaMemcpy(cgah, goaheadnodeofcomponent, cs.nelements * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
assert(cudaMemcpy(cminwt, minwtcomponent, cs.nelements * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
assert(cudaMemcpy(cpart, partners, cs.nelements * sizeof(unsigned), cudaMemcpyDeviceToHost) == cudaSuccess);
assert(cudaMemcpy(cpin, pin, cs.nelements * sizeof(bool), cudaMemcpyDeviceToHost) == cudaSuccess);

for(int i = 0; i < cs.nelements; i++)
{
if(ele2comp[i] == i && cminwt[i] != MYINFINITY && cpin[cgah[i]])
printf("CM %d %d %d %d\n",  i, cminwt[i], cgah[i], cpart[i]);
}

free(ele2comp);
free(cgah);
free(cminwt);
}

__global__ void dfindcompmintwo_serial(unsigned *mstwt, Graph graph, ComponentSpace csr, ComponentSpace csw, DistT *eleminwts, DistT *minwtcomponent, unsigned *partners, unsigned *phores, bool *processinnextiteration, unsigned *goaheadnodeofcomponent, unsigned inpid, GlobalBarrier gb, bool *repeat, unsigned *count) {
unsigned id;
if (inpid < graph.nnodes) id = inpid;

unsigned srcboss, dstboss;

if(id < graph.nnodes && processinnextiteration[id])
{
srcboss = csw.find(id);
dstboss = csw.find(partners[id]);
}

gb.Sync();

if (id < graph.nnodes && processinnextiteration[id] && srcboss != dstboss) {
dprintf("trying unify id=%d (%d -> %d)\n", id, srcboss, dstboss);

if (csw.unify(srcboss, dstboss)) {
atomicAdd(mstwt, eleminwts[id]);
dprintf("u %d -> %d (%d)\n", srcboss, dstboss, eleminwts[id]);
processinnextiteration[id] = false;
eleminwts[id] = MYINFINITY;	
}
else {
*repeat = true;
}

dprintf("\tcomp[%d] = %d.\n", srcboss, csw.find(srcboss));
}

gb.Sync(); 
}
