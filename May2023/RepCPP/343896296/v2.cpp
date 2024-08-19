

#include "v2.hpp"



#define BLOCK_SIZE 9
#define BLOCK_HEIGHT 3
#define BLOCK_WIDTH 3

using namespace std;


void V2::Execute(Runtime rt){   

if(!rt.opt_csr_a || rt.opt_csr_b || !rt.opt_csr_f){
printf("[Error] Flags '--opt-csr-a' and '--opt-csr-f' are required for V2.\n");
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

Block9Permutations permute = Block9Permutations();
permute.Permutate(rt.threads);



Block9 block;
COOMatrix coo;

int tmpnnz = 0;


#pragma omp parallel for \
shared(A,B,F,C,tmpnnz) \
private(block,coo)
for(int i=0; i<A->H; i+=BLOCK_HEIGHT){      

coo.Reset();

for(int j=0; j<A->W; j+=BLOCK_WIDTH){   

block.UpdateBlockPosition(i, j);
block.Reset();
block.value = 0;


for(int k=0; k<A->H; k+=BLOCK_HEIGHT){

if (block.isAllOnes() ){
break;
}

block.BlockOR( 
permute.GetPermutation(
CSCBlocking9::GetBlockValue(A, k, i),
CSCBlocking9::GetBlockValue(B, k, j)
)
);

}



CSCBlocking9::AddCOOfromBlockValue(&coo, block.value, i, j);

}

#pragma omp critical
{
C->coo.insert( C->coo.end(), coo.coo.begin(), coo.coo.end() );
C->nnz += coo.nnz;
}
}


clock_gettime(CLOCK_MONOTONIC_RAW, &end);
float delta_us = (float) ((end.tv_sec - start.tv_sec) * 1000000 + (end.tv_nsec - start.tv_nsec) / 1000)/ (1000000);
printf("[Info] V2 took %f s\n", delta_us);

}

#undef BLOCK_SIZE
#undef BLOCK_HEIGHT
#undef BLOCK_WIDTH
