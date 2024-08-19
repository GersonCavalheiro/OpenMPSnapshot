

#include "v4.hpp"



#define BLOCK_SIZE 64
#define BLOCK_HEIGHT 8
#define BLOCK_WIDTH 8

using namespace std;


void V4::Execute(Runtime rt){   

if(!rt.opt_csr_b || rt.opt_csr_a || rt.opt_csr_f){
printf("[Error] Flag '--opt-csr-b' is required for V4.\n");
exit(EXIT_FAILURE);
}

time_t tim;
srand((unsigned) time(&tim));
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC_RAW, &start);

MPIUtil mpi = MPIUtil();


CSCMatrix* A = rt.A;    
CSCMatrix* B = rt.B;    
CSCMatrix* F = rt.F;    
COOMatrix* C = new COOMatrix(); 

if(A->W != B->H || A->H != B->W || F->H != B->W){
printf("[Error] Matrix Sizes do not match\n");
exit(EXIT_FAILURE);
}

Noodle noodleA = Noodle();
noodleA.LoadNoodleFromCSC(A, 0);

Noodle noodleB = Noodle();
noodleB.LoadNoodleFromCSC(B, 0);

Noodle noodleF = Noodle();
noodleF.LoadNoodleFromCSC(F, 0);

Block64 block;
COOMatrix coo;

int pStart = mpi.NodeIdx() * BLOCK_HEIGHT * (int)((A->H / BLOCK_HEIGHT) / mpi.ClusterSize());

int pEnd = pStart + BLOCK_HEIGHT * (int)((A->H / BLOCK_HEIGHT ) / mpi.ClusterSize()) - 1;

if(mpi.NodeIdx()==mpi.ClusterSize()-1)
pEnd = A->H -1;

#ifdef DEBUGGING_CHECKS
printf("[Info] Batch for MPI node %d: %d..%d\n", mpi.NodeIdx(), pStart, pEnd);
#endif


#pragma omp parallel for \
shared(noodleA,noodleB,noodleF,C) \
private(block,coo) 
for(int i=pStart; i<=pEnd; i+=BLOCK_HEIGHT){      

coo.Reset();

for(int j=0; j<A->W; j+=BLOCK_WIDTH){   
block.UpdateBlockPosition(i, j);
block.Reset();


for(int k=0; k<A->H; k+=BLOCK_HEIGHT){

if (block.isAllOnes())
break;

block.BlockOR( 
CSCBlocking64::MultiplyBlocks(
CSCBlocking64::GetBlockValue(&noodleA, i, k),
CSCBlocking64::GetBlockValue(&noodleB, j, k),
(uint64_t)0  
)
);

}


CSCBlocking64::AddCOOfromBlockValue(&coo, block.value, i, j);

}

#pragma omp critical
{
C->coo.insert( C->coo.end(), coo.coo.begin(), coo.coo.end() );
C->nnz += coo.nnz;
}

}


delete A;
delete B;
delete F;

if(mpi.IsMaster()){ 

for(int i=1; i<mpi.ClusterSize(); i++){

MPI_Request* mpiReq = new MPI_Request();

uint32_t* buffNNZ;

mpi.Receive_t(
MPI_TAG_SEND_NNZ,
buffNNZ,
2,
MPI_ANY_SOURCE,
MPI_UNSIGNED,
mpiReq
);
mpi.ReceiveWait(mpiReq);

uint32_t nnz  = buffNNZ[0];
uint32_t node = buffNNZ[1];

if(nnz < 0 || node < 1 || node >= mpi.ClusterSize()){
printf("[Error] NNZ or Node received by naster was wrong (node: %d)\n", node);
mpi.Abort();
exit(EXIT_FAILURE);
}

if(nnz<=0)
continue;

uint32_t * buffCOO = (uint32_t*) malloc(sizeof(uint32_t) * nnz * 2);
uint32_t end = nnz;

mpi.Receive_t(
MPI_TAG_SEND_COO,
buffCOO,
nnz * 2,
node,
MPI_UNSIGNED,
mpiReq
);
mpi.ReceiveWait(mpiReq);

C->coo.reserve(C->nnz + nnz);
for(int i=0; i<nnz; i++){
C->coo.emplace_back(buffCOO[i], buffCOO[i+end]);
}
C->nnz += nnz;

free(buffCOO);
free(mpiReq);

}

}else{  


MPI_Request* mpiReq = new MPI_Request();

uint32_t buffNNZ[] = {(uint32_t)C->nnz, (uint32_t)mpi.NodeIdx()};
mpi.Send_t(
MPI_TAG_SEND_NNZ,
buffNNZ,
2,
MPI_MASTER_NODE_IDX,
MPI_UNSIGNED,
mpiReq
);

if(C->nnz<=0){
mpi.SendWait(mpiReq);
}else{

uint32_t * buffCOO = (uint32_t*) malloc(sizeof(uint32_t) * C->nnz * 2);
uint32_t end = C->nnz;
for(int i=0; i<C->nnz; i++){
buffCOO[    i] = C->coo[i].first;
buffCOO[end+i] = C->coo[i].second;
}

mpi.SendWait(mpiReq);

mpi.Send_t(
MPI_TAG_SEND_COO,
buffCOO,
C->nnz * 2,
MPI_MASTER_NODE_IDX,
MPI_UNSIGNED,
mpiReq
);
mpi.SendWait(mpiReq);

free(buffCOO);

}

}

MPI_Barrier(MPI_COMM_WORLD);

if(mpi.IsMaster()){
clock_gettime(CLOCK_MONOTONIC_RAW, &end);
float delta_us = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000)/ (1000000);
printf("[Info] V4 took %f s\n", delta_us);


mpi.Finalize();
}


delete C;

return;

}

#undef BLOCK_SIZE
#undef BLOCK_HEIGHT
#undef BLOCK_WIDTH
