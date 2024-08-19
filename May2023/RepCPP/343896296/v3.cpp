

#include "v3.hpp"



#define BLOCK_SIZE 64
#define BLOCK_HEIGHT 8
#define BLOCK_WIDTH 8

using namespace std;


void V3::Execute(Runtime rt){   

if(!rt.opt_csr_b || rt.opt_csr_a || rt.opt_csr_f){
printf("[Error] Flag '--opt-csr-b' is required for V3.\n");
exit(EXIT_FAILURE);
}

time_t tim;
srand((unsigned) time(&tim));
struct timespec start, end;
clock_gettime(CLOCK_MONOTONIC_RAW, &start);

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

#pragma omp parallel for \
shared(noodleA,noodleB,noodleF,C) \
private(block,coo) 
for(int i=0; i<A->H; i+=BLOCK_HEIGHT){      

coo.Reset();

for(int j=0; j<A->W; j+=BLOCK_WIDTH){   

block.UpdateBlockPosition(i, j);
block.Reset();


for(int k=0; k<A->H; k+=BLOCK_HEIGHT){

if (block.isAllOnes() ){
break;
}

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


clock_gettime(CLOCK_MONOTONIC_RAW, &end);
float delta_us = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000)/ (1000000);
printf("[Info] V3 took %f s\n", delta_us);

}

#undef BLOCK_SIZE
#undef BLOCK_HEIGHT
#undef BLOCK_WIDTH
