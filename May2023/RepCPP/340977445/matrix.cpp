#include "utils.hpp"
#include "timer.hpp"
#include <iostream>
#include <random>
#include <vector>
#include <cassert>
#include <omp.h>
#include <memory>
#include <cstring>
#include <array>

#include <cblas.h>

typedef double decimal;


int main(int argc, char** argv){
if(argc != 2 || (argc == 2 && (strcmp(argv[1], "-check") !=0 && strcmp(argv[1], "-no-check") && strcmp(argv[1], "-no-optim")))){
std::cout << "Should be\n";
std::cout << argv[0] << " -check #to check the result\n";
std::cout << argv[0] << " -no-check #to avoid checking the result\n";   
std::cout << argv[0] << " -no-optim #to avoid running the optimized code\n";  
return 1;     
}

const bool checkRes = (strcmp(argv[1], "-no-check") != 0);
const bool runOptim = (strcmp(argv[1], "-no-optim") != 0);

const long int N = 1024;        
decimal* A = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
memset(A, 0, N * N * sizeof(decimal));
decimal* B = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
memset(B, 0, N * N * sizeof(decimal));
decimal* C = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
memset(C, 0, N * N * sizeof(decimal));
decimal* COptim = (decimal*)aligned_alloc(64, N * N * sizeof(decimal));
memset(COptim, 0, N * N * sizeof(decimal));

{        
std::mt19937 gen(0);
std::uniform_real_distribution<decimal> dis(0, 1);

for(long int i = 0 ; i < N ; ++i){
for(long int j = 0 ; j < N ; ++j){
A[i*N+j] = dis(gen);
B[j*N+i] = dis(gen);
}
}
}   

Timer timerNoOptim;
if(checkRes){
for(long int k = 0 ; k < N ; ++k){
for(long int j = 0 ; j < N ; ++j){
for(long int i = 0 ; i < N ; ++i){
C[i*N+j] += A[i*N+k] * B[j*N+k];
}
}
}
}
timerNoOptim.stop();

Timer timerWithOptim;
if(runOptim){











decimal* ptr1 = &A[0];
decimal* ptr2 = &B[0];
#pragma omp parallel for collapse(1)
for(long int i = 0 ; i < N ; ++i){
for(long int j = 0 ; j < N ; ++j){
decimal sum = 0;
#pragma omp simd reduction(+: sum) aligned(ptr1, ptr2: 64) safelen(N) simdlen(64)
for(long int k = 0 ; k < N ; ++k){
sum += *(ptr1 + i*N+k) * *(ptr2 + j*N+k);
}
COptim[i*N+j] = sum;
}
}




}
timerWithOptim.stop();

if(checkRes){
std::cout << ">> Without Optim : " << timerNoOptim.getElapsed() << std::endl;
if(runOptim){
for(long int i = 0 ; i < N ; ++i){
for(long int j = 0 ; j < N ; ++j){
CheckEqual(C[i*N+j],COptim[i*N+j]);
}
}
}
}
if(runOptim){
std::cout << ">> With Optim : " << timerWithOptim.getElapsed() << std::endl;
}

free(A);
free(B);
free(C);
free(COptim);

return 0;
}