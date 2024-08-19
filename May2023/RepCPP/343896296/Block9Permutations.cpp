

#include "Block9Permutations.hpp"


Block9Permutations::Block9Permutations(){

#if (PERM_BLOCK_SIZE != 9)
printf("[Error] Block size is not 9.\n");
exit(EXIT_FAILURE);
#endif    

int pr = 1; 
int upper = (int)pow(2, PERM_BLOCK_SIZE);
if(upper > 1){
for(int i=upper-1; i<=upper; i++){
pr = pr * i;
}
}

permutations.resize(pr+1); 
}

int Block9Permutations::block_data_matrix_mult(int L, int R){

bool aL = (L >> 8)&1;
bool bL = (L >> 7)&1;
bool cL = (L >> 6)&1;
bool dL = (L >> 5)&1;
bool eL = (L >> 4)&1;
bool fL = (L >> 3)&1;
bool gL = (L >> 2)&1;
bool hL = (L >> 1)&1;
bool iL = (L     )&1;

bool aR = (R >> 8)&1;
bool bR = (R >> 7)&1;
bool cR = (R >> 6)&1;
bool dR = (R >> 5)&1;
bool eR = (R >> 4)&1;
bool fR = (R >> 3)&1;
bool gR = (R >> 2)&1;
bool hR = (R >> 1)&1;
bool iR = (R     )&1;

bool a = aL&aR | bL&bR | cL&cR ;
bool d = aL&dR | bL&eR | cL&fR ;
bool g = aL&gR | bL&hR | cL&iR ;
bool b = dL&aR | eL&bR | fL&cR ;
bool e = dL&dR | eL&eR | fL&fR ;
bool h = dL&gR | eL&hR | fL&iR ;
bool c = gL&aR | hL&bR | iL&cR ;
bool f = gL&dR | hL&eR | iL&fR ;
bool i = gL&gR | hL&hR | iL&iR ;

return (int)(
(a << 8) | (b << 7) | (c << 6) | (d << 5) |
(e << 4) | (f << 3) | (g << 2) | (h << 1) | int(i) 
);
}

int Block9Permutations::PermutationIdx(int leftState, int rightState){
return leftState * ((int) pow(2, PERM_BLOCK_SIZE) -1) + rightState; 
}

void Block9Permutations::permutate_thread(int LSideBegin, int LSideEnd){

int RSideEnd = (int)pow(2, PERM_BLOCK_SIZE) -1;

for (int i=LSideBegin; i<=LSideEnd; i++){
for (int j=0; j<RSideEnd; j++){

this->permutations.at(PermutationIdx(i,j)) = block_data_matrix_mult(i, j);

}
}

}


void Block9Permutations::Permutate(int threads){


#pragma omp parallel for
for(int i=0; i<threads; i++){

int LSideBegin = i * pow(2, PERM_BLOCK_SIZE) / threads;
int LSideEnd = (i+1) * pow(2, PERM_BLOCK_SIZE) / threads -1;

#pragma omp task
permutate_thread(LSideBegin, LSideEnd);
}

}

int Block9Permutations::GetPermutationAt(int permutationIdx){
return permutations.at(permutationIdx);
}

int Block9Permutations::GetPermutation(int leftState, int rightState){
return permutations.at(
PermutationIdx(leftState, rightState)
);
}






